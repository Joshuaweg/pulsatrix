#include "pulsatrix/protein_contacts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/variant_scoring.hpp"  // ScoringPassBytes

namespace pulsatrix {

EsmContactHead LoadEsmContactHead(const std::string& directory, const EncoderLMConfig& config) {
    const HfCheckpoint checkpoint = HfCheckpoint::Open(directory);
    for (const std::string prefix : {"esm.contact_head.regression.", "contact_head.regression."}) {
        if (!checkpoint.contains(prefix + "weight")) continue;
        CPUBackend host;
        EsmContactHead head{config.num_hidden_layers, config.num_attention_heads,
                            checkpoint.tensor(prefix + "weight", &host).to_host_vector(), 0.0f};
        if (static_cast<int64_t>(head.weight.size()) != head.layers * head.heads) {
            throw std::invalid_argument("LoadEsmContactHead: " + prefix + "weight has " + std::to_string(head.weight.size()) +
                                        " values, not one per head (" + std::to_string(head.layers * head.heads) + ")");
        }
        if (checkpoint.contains(prefix + "bias")) {
            const std::vector<float> b = checkpoint.tensor(prefix + "bias", &host).to_host_vector();
            if (b.size() != 1) throw std::invalid_argument("LoadEsmContactHead: " + prefix + "bias isn't one value");
            head.bias = b[0];
        }
        return head;
    }
    throw std::invalid_argument("LoadEsmContactHead: " + directory + " has no contact head");
}

ContactMap ContactFeatures(const float* attention, int64_t tokens) {
    if (tokens < 3) throw std::invalid_argument("ContactFeatures: needs <cls>, at least one residue and <eos>");
    const int64_t L = tokens - 2;
    // The residue block (rows and columns 1..L), symmetrized; sums in double.
    std::vector<double> f(static_cast<size_t>(L * L));
    std::vector<double> row(static_cast<size_t>(L), 0.0);
    double total = 0;
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = 0; j < L; ++j) {
            const double v = static_cast<double>(attention[(i + 1) * tokens + j + 1]) + attention[(j + 1) * tokens + i + 1];
            f[static_cast<size_t>(i * L + j)] = v;
            row[static_cast<size_t>(i)] += v;
            total += v;
        }
    }
    // Symmetric, so column sums equal row sums.
    ContactMap out{L, std::vector<float>(static_cast<size_t>(L * L))};
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = 0; j < L; ++j) {
            const size_t k = static_cast<size_t>(i * L + j);
            const double apc = total == 0 ? 0.0 : row[static_cast<size_t>(i)] * row[static_cast<size_t>(j)] / total;
            out.values[k] = static_cast<float>(f[k] - apc);
        }
    }
    return out;
}

ContactPredictor::ContactPredictor(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend, ContactOptions options)
    : model_(model), tokenizer_(tokenizer), backend_(backend), options_(options) {
    for (const char* t : {"<cls>", "<eos>"}) {
        if (!tokenizer.token_to_id(t)) throw std::invalid_argument(std::string("ContactPredictor: the tokenizer has no ") + t);
    }
    if (options_.max_pass_bytes < 1) throw std::invalid_argument("ContactPredictor: max_pass_bytes must be at least 1");
}

void ContactPredictor::Run(std::string_view sequence, const std::function<void(int64_t, const std::vector<float>&, int64_t)>& on_layer) {
    if (sequence.empty()) throw std::invalid_argument("ContactPredictor: the sequence is empty");
    std::vector<float> ids{static_cast<float>(*tokenizer_.token_to_id("<cls>"))};
    for (char c : sequence) {
        const auto id = tokenizer_.token_to_id(std::string(1, c));
        if (!id) throw std::invalid_argument(std::string("ContactPredictor: '") + c + "' isn't a vocabulary token");
        ids.push_back(static_cast<float>(*id));
    }
    ids.push_back(static_cast<float>(*tokenizer_.token_to_id("<eos>")));
    const auto T = static_cast<int64_t>(ids.size());
    const int64_t bytes = ScoringPassBytes(model_.config(), T);
    if (bytes > options_.max_pass_bytes) {
        throw std::invalid_argument("ContactPredictor: a " + std::to_string(T) + "-token pass needs about " + std::to_string(bytes >> 20) +
                                    " MiB, over max_pass_bytes (" + std::to_string(options_.max_pass_bytes >> 20) + " MiB)");
    }
    model_.clear_padding_mask();
    const bool keep = model_.keep_activations();
    model_.set_keep_activations(false);
    model_.set_attention_observer([&](int64_t layer, const Tensor& attention) { on_layer(layer, attention.to_host_vector(), T); });
    try {
        (void)model_.forward(Tensor(Shape({1, T}), backend_, ids, backend_->device()));
    } catch (...) {
        model_.set_attention_observer({});
        model_.set_keep_activations(keep);
        throw;
    }
    model_.set_attention_observer({});
    model_.set_keep_activations(keep);
}

ContactMap ContactPredictor::predict(std::string_view sequence, const EsmContactHead& head) {
    const int64_t H = model_.config().num_attention_heads;
    if (head.layers != model_.num_layers() || head.heads != H || static_cast<int64_t>(head.weight.size()) != head.layers * H) {
        throw std::invalid_argument("ContactPredictor::predict: the contact head doesn't fit the model's layers and heads");
    }
    const auto L = static_cast<int64_t>(sequence.size());
    std::vector<double> logit(static_cast<size_t>(L * L), head.bias);
    Run(sequence, [&](int64_t layer, const std::vector<float>& attention, int64_t T) {
        for (int64_t h = 0; h < H; ++h) {
            const ContactMap f = ContactFeatures(attention.data() + h * T * T, T);
            const double w = head.weight[static_cast<size_t>(layer * H + h)];
            for (size_t k = 0; k < logit.size(); ++k) logit[k] += w * f.values[k];
        }
    });
    ContactMap out{L, std::vector<float>(logit.size())};
    for (size_t k = 0; k < logit.size(); ++k) out.values[k] = static_cast<float>(1.0 / (1.0 + std::exp(-logit[k])));
    return out;
}

ContactMap ContactPredictor::average_heads(std::string_view sequence, const std::vector<AttentionHead>& heads) {
    if (heads.empty()) throw std::invalid_argument("ContactPredictor::average_heads: no heads");
    const int64_t H = model_.config().num_attention_heads;
    for (const AttentionHead& a : heads) {
        if (a.layer < 0 || a.layer >= model_.num_layers() || a.head < 0 || a.head >= H) {
            throw std::invalid_argument("ContactPredictor::average_heads: head " + std::to_string(a.layer) + "." + std::to_string(a.head) +
                                        " isn't in the model");
        }
    }
    const auto L = static_cast<int64_t>(sequence.size());
    std::vector<double> sum(static_cast<size_t>(L * L), 0.0);
    Run(sequence, [&](int64_t layer, const std::vector<float>& attention, int64_t T) {
        for (const AttentionHead& a : heads) {
            if (a.layer != layer) continue;
            const ContactMap f = ContactFeatures(attention.data() + a.head * T * T, T);
            for (size_t k = 0; k < sum.size(); ++k) sum[k] += f.values[k];
        }
    });
    ContactMap out{L, std::vector<float>(sum.size())};
    for (size_t k = 0; k < sum.size(); ++k) out.values[k] = static_cast<float>(sum[k] / static_cast<double>(heads.size()));
    return out;
}

void ContactPredictor::for_each_head(std::string_view sequence, const std::function<void(AttentionHead, const ContactMap&)>& visit) {
    const int64_t H = model_.config().num_attention_heads;
    Run(sequence, [&](int64_t layer, const std::vector<float>& attention, int64_t T) {
        for (int64_t h = 0; h < H; ++h) visit({layer, h}, ContactFeatures(attention.data() + h * T * T, T));
    });
}

double ContactPrecision(const ContactMap& predicted, const ContactMap& truth, SeparationRange range, int64_t top) {
    if (predicted.length != truth.length || predicted.values.size() != truth.values.size() ||
        predicted.values.size() != static_cast<size_t>(predicted.length * predicted.length)) {
        throw std::invalid_argument("ContactPrecision: the predicted and true maps must both be L x L");
    }
    if (top < 0) throw std::invalid_argument("ContactPrecision: top must not be negative");
    if (top == 0) return std::numeric_limits<double>::quiet_NaN();
    const int64_t L = predicted.length;
    std::vector<std::pair<float, float>> pairs;  // (score, truth), in (i, j) order
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = i + std::max<int64_t>(range.min, 1); j < L; ++j) {
            if (range.max > 0 && j - i >= range.max) break;
            const float t = truth.at(i, j);
            if (!std::isnan(t)) pairs.emplace_back(predicted.at(i, j), t);
        }
    }
    const auto k = std::min<size_t>(static_cast<size_t>(top), pairs.size());
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    double hits = 0;
    for (size_t n = 0; n < k; ++n) hits += pairs[n].second > 0.5f ? 1.0 : 0.0;
    return hits / static_cast<double>(top);
}

ContactPrecisions PrecisionAtL(const ContactMap& predicted, const ContactMap& truth, SeparationRange range) {
    const int64_t L = predicted.length;
    return {ContactPrecision(predicted, truth, range, L), ContactPrecision(predicted, truth, range, L / 2),
            ContactPrecision(predicted, truth, range, L / 5)};
}

ContactEvaluation EvaluateContacts(const ContactMap& predicted, const ContactMap& truth) {
    return {PrecisionAtL(predicted, truth, kShortRange), PrecisionAtL(predicted, truth, kMediumRange),
            PrecisionAtL(predicted, truth, kLongRange)};
}

std::vector<HeadPrecision> RankContactHeads(ContactPredictor& predictor, const std::vector<LabeledProtein>& proteins, SeparationRange range) {
    if (proteins.empty()) throw std::invalid_argument("RankContactHeads: no proteins");
    std::vector<HeadPrecision> ranked;
    for (size_t p = 0; p < proteins.size(); ++p) {
        const LabeledProtein& protein = proteins[p];
        const auto L = static_cast<int64_t>(protein.sequence.size());
        if (protein.contacts.length != L || protein.contacts.values.size() != static_cast<size_t>(L * L)) {
            throw std::invalid_argument("RankContactHeads: protein " + std::to_string(p) + "'s contact map isn't L x L for its sequence");
        }
        size_t h = 0;
        predictor.for_each_head(protein.sequence, [&](AttentionHead head, const ContactMap& features) {
            if (p == 0) ranked.push_back({head, 0.0});
            ranked[h++].precision += ContactPrecision(features, protein.contacts, range, L) / static_cast<double>(proteins.size());
        });
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const HeadPrecision& a, const HeadPrecision& b) { return a.precision > b.precision; });
    return ranked;
}

}  // namespace pulsatrix
