#include "pulsatrix/protein_training.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "portable_random.hpp"
#include "pulsatrix/variant_scoring.hpp"  // kAminoAcids

namespace pulsatrix {

namespace {

uint64_t SplitMix(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int64_t Id(const TextTokenizer& tok, const std::string& token, const char* who) {
    const auto id = tok.token_to_id(token);
    if (!id) throw std::invalid_argument(std::string(who) + ": the tokenizer has no " + token);
    return *id;
}

Tensor Reshaped(const Tensor& t, Shape shape) {
    Tensor out(t);
    out.reshape(std::move(shape));
    return out;
}

}  // namespace

// ---- collator ---------------------------------------------------------------------------------

MaskedLMCollator::MaskedLMCollator(const TextTokenizer& tokenizer, DeviceBackend* backend, MaskedLMOptions options, uint64_t seed)
    : tokenizer_(tokenizer), backend_(backend), options_(options), state_(seed) {
    const char* who = "MaskedLMCollator";
    cls_ = Id(tokenizer, "<cls>", who);
    eos_ = Id(tokenizer, "<eos>", who);
    pad_ = Id(tokenizer, "<pad>", who);
    mask_ = Id(tokenizer, "<mask>", who);
    for (const char* a = kAminoAcids; *a != 0; ++a) amino_acids_.push_back(Id(tokenizer, std::string(1, *a), who));
    auto in01 = [](double p) { return p >= 0.0 && p <= 1.0; };
    if (!in01(options.mask_probability) || !in01(options.mask_fraction) || !in01(options.random_fraction) ||
        options.mask_fraction + options.random_fraction > 1.0 + 1e-12 || options.max_tokens < 3) {
        throw std::invalid_argument("MaskedLMCollator: probabilities must be in [0, 1], mask_fraction + random_fraction at most 1, "
                                    "and max_tokens at least 3");
    }
}

uint64_t MaskedLMCollator::Next() { return SplitMix(state_); }
double MaskedLMCollator::Uniform() { return (static_cast<double>(Next() >> 11) + 0.5) / static_cast<double>(1ULL << 53); }

std::vector<int64_t> MaskedLMCollator::tokens(const std::string& sequence) {
    if (sequence.empty()) throw std::invalid_argument("MaskedLMCollator: a sequence is empty");
    const auto max_residues = static_cast<size_t>(options_.max_tokens - 2);
    size_t start = 0, length = sequence.size();
    if (length > max_residues) {
        start = static_cast<size_t>(Next() % (sequence.size() - max_residues + 1));
        length = max_residues;
    }
    std::vector<int64_t> ids{cls_};
    for (size_t i = start; i < start + length; ++i) {
        const auto id = tokenizer_.token_to_id(std::string(1, sequence[i]));
        if (!id) throw std::invalid_argument(std::string("MaskedLMCollator: '") + sequence[i] + "' isn't a vocabulary token");
        ids.push_back(*id);
    }
    ids.push_back(eos_);
    return ids;
}

MaskedLMBatch MaskedLMCollator::collate(const std::vector<std::string>& sequences) {
    if (sequences.empty()) throw std::invalid_argument("MaskedLMCollator: no sequences");
    std::vector<std::vector<int64_t>> rows;
    size_t L = 0;
    for (const std::string& s : sequences) {
        rows.push_back(tokens(s));
        L = std::max(L, rows.back().size());
    }
    const auto N = static_cast<int64_t>(rows.size());
    std::vector<float> ids(static_cast<size_t>(N) * L, static_cast<float>(pad_)), keep(ids.size(), 0.0f),
        targets(ids.size(), static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex));
    int64_t num_targets = 0;
    for (int64_t n = 0; n < N; ++n) {
        const std::vector<int64_t>& row = rows[static_cast<size_t>(n)];
        for (size_t l = 0; l < row.size(); ++l) {
            const size_t k = static_cast<size_t>(n) * L + l;
            keep[k] = 1.0f;
            ids[k] = static_cast<float>(row[l]);
            if (l == 0 || l + 1 == row.size()) continue;  // <cls>, <eos>
            if (Uniform() >= options_.mask_probability) continue;
            targets[k] = static_cast<float>(row[l]);
            ++num_targets;
            const double r = Uniform();
            if (r < options_.mask_fraction) {
                ids[k] = static_cast<float>(mask_);
            } else if (r < options_.mask_fraction + options_.random_fraction) {
                ids[k] = static_cast<float>(amino_acids_[static_cast<size_t>(Next() % amino_acids_.size())]);
            }
        }
    }
    const Shape shape({N, static_cast<int64_t>(L)});
    return MaskedLMBatch{Tensor(shape, backend_, ids, backend_->device()), Tensor(shape, backend_, keep, backend_->device()),
                         Tensor(shape, backend_, targets, backend_->device()), num_targets};
}

// ---- sampler ----------------------------------------------------------------------------------

ClusterSampler::ClusterSampler(const std::vector<std::string>& cluster_of, uint64_t seed) : state_(seed) {
    if (cluster_of.empty()) throw std::invalid_argument("ClusterSampler: no sequences");
    std::map<std::string, size_t> index;
    for (size_t i = 0; i < cluster_of.size(); ++i) {
        const auto [it, added] = index.emplace(cluster_of[i], members_.size());
        if (added) members_.emplace_back();
        members_[it->second].push_back(static_cast<int64_t>(i));
    }
}

int64_t ClusterSampler::next() {
    const std::vector<int64_t>& cluster = members_[static_cast<size_t>(SplitMix(state_) % members_.size())];
    return cluster[static_cast<size_t>(SplitMix(state_) % cluster.size())];
}

// ---- initialization, training and evaluation ------------------------------------------------

void InitializeEsm(EncoderLM& model, uint64_t seed, float std) {
    PortableRng rng{seed};
    const int64_t pad = model.config().pad_token_id, h = model.config().hidden_size;
    for (const NamedParamRef& p : model.named_parameters()) {
        Tensor& t = *p.ref.value;
        std::vector<float> v(static_cast<size_t>(t.numel()));
        const bool matrix = t.rank() >= 2;
        const bool norm_gain = !matrix && p.name.find("norm") != std::string::npos &&
                               (p.name.size() >= 6 && p.name.compare(p.name.size() - 6, 6, "weight") == 0);
        if (matrix) {
            for (float& x : v) x = static_cast<float>(std * rng.gaussian());
            if (p.name == "embed_tokens.weight" && pad >= 0 && pad < model.config().vocab_size) {
                std::fill_n(v.begin() + pad * h, h, 0.0f);
            }
        } else if (norm_gain) {
            std::fill(v.begin(), v.end(), 1.0f);
        }  // biases and norm offsets: 0
        t = Tensor(t.shape(), t.backend(), v, t.device());
    }
}

float MaskedLMForwardBackward(EncoderLM& model, const MaskedLMBatch& batch, TokenCrossEntropyLoss& loss, float normalizer) {
    model.set_padding_mask(batch.keep);
    const Tensor logits = model.forward(batch.ids);
    const float value = loss.forward(logits, batch.targets, normalizer);
    (void)model.backward(loss.backward());
    return value;
}

MaskedLMEvaluation EvaluateMaskedLM(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend,
                                    const std::vector<std::string>& sequences, int64_t batch_size, uint64_t seed, MaskedLMOptions options) {
    if (batch_size < 1) throw std::invalid_argument("EvaluateMaskedLM: batch_size must be at least 1");
    MaskedLMCollator collator(tokenizer, backend, options, seed);
    const bool keep = model.keep_activations();
    model.set_keep_activations(false);
    const int64_t V = model.config().vocab_size;
    double nll = 0;
    int64_t correct = 0, count = 0;
    try {
        for (size_t start = 0; start < sequences.size(); start += static_cast<size_t>(batch_size)) {
            const std::vector<std::string> chunk(sequences.begin() + static_cast<std::ptrdiff_t>(start),
                                                 sequences.begin() + static_cast<std::ptrdiff_t>(std::min(sequences.size(), start + batch_size)));
            const MaskedLMBatch b = collator.collate(chunk);
            model.set_padding_mask(b.keep);
            const std::vector<float> logits = model.forward(b.ids).to_host_vector(), targets = b.targets.to_host_vector();
            for (size_t t = 0; t < targets.size(); ++t) {
                if (targets[t] < 0) continue;
                const float* row = logits.data() + t * static_cast<size_t>(V);
                const float top = *std::max_element(row, row + V);
                double z = 0;
                for (int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v] - top));
                nll -= static_cast<double>(row[static_cast<int64_t>(targets[t])] - top) - std::log(z);
                correct += std::max_element(row, row + V) - row == static_cast<int64_t>(targets[t]) ? 1 : 0;
                ++count;
            }
        }
    } catch (...) {
        model.clear_padding_mask();
        model.set_keep_activations(keep);
        throw;
    }
    model.clear_padding_mask();
    model.set_keep_activations(keep);
    MaskedLMEvaluation e;
    e.tokens = count;
    if (count > 0) {
        e.loss = nll / static_cast<double>(count);
        e.perplexity = std::exp(e.loss);
        e.accuracy = static_cast<double>(correct) / static_cast<double>(count);
    }
    return e;
}

// ---- heads ------------------------------------------------------------------------------------

SequenceHead::SequenceHead(int64_t hidden_size, int64_t outputs, Pooling pooling, DeviceBackend* backend, uint64_t seed)
    : hidden_(hidden_size), outputs_(outputs), pooling_(pooling), backend_(backend), linear_(hidden_size, outputs, backend) {
    if (hidden_size < 1 || outputs < 1) throw std::invalid_argument("SequenceHead: hidden_size and outputs must be positive");
    PortableRng rng{seed};
    std::vector<float> w(static_cast<size_t>(hidden_size * outputs));
    for (float& x : w) x = static_cast<float>(0.02 * rng.gaussian());
    linear_.set_weight(w);
}

void SequenceHead::set_residue_mask(const Tensor& residues) {
    if (residues.rank() != 2) throw std::invalid_argument("SequenceHead::set_residue_mask: the mask must be (N, L)");
    residues_ = residues.to_host_vector();
    residues_shape_ = residues.shape();
}

Tensor SequenceHead::forward_impl(const Tensor& input) {
    if (input.rank() != 3 || input.shape().dim(2) != hidden_) throw std::invalid_argument("SequenceHead::forward: input must be (N, L, hidden)");
    const int64_t N = input.shape().dim(0), L = input.shape().dim(1), h = hidden_;
    if (pooling_ == Pooling::PerResidue) {
        last_shape_ = input.shape();
        return Reshaped(linear_.forward(Reshaped(input, Shape({N * L, h}))), Shape({N, L, outputs_}));
    }
    if (residues_shape_ != Shape({N, L})) throw std::invalid_argument("SequenceHead::forward: set a residue mask of the input's (N, L)");
    last_input_ = input.to_host_vector();
    last_shape_ = input.shape();
    std::vector<float> pooled(static_cast<size_t>(N * h), 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        double count = 0;
        for (int64_t l = 0; l < L; ++l) count += residues_[static_cast<size_t>(n * L + l)];
        if (count == 0) throw std::invalid_argument("SequenceHead::forward: a sequence has no residues in the mask");
        for (int64_t l = 0; l < L; ++l) {
            const float m = residues_[static_cast<size_t>(n * L + l)];
            if (m == 0.0f) continue;
            for (int64_t j = 0; j < h; ++j) pooled[static_cast<size_t>(n * h + j)] += static_cast<float>(m * last_input_[static_cast<size_t>((n * L + l) * h + j)] / count);
        }
    }
    return linear_.forward(Tensor(Shape({N, h}), backend_, pooled, backend_->device()));
}

Tensor SequenceHead::backward(const Tensor& grad_output) {
    if (last_shape_.rank() != 3) throw std::logic_error("SequenceHead::backward: called before any forward()");
    if (pooling_ == Pooling::PerResidue) {
        const int64_t rows = last_shape_.dim(0) * last_shape_.dim(1);
        return Reshaped(linear_.backward(Reshaped(grad_output, Shape({rows, outputs_}))), last_shape_);
    }
    const int64_t N = last_shape_.dim(0), L = last_shape_.dim(1), h = hidden_;
    const std::vector<float> g = linear_.backward(grad_output).to_host_vector();  // (N, h)
    std::vector<float> out(static_cast<size_t>(N * L * h), 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        double count = 0;
        for (int64_t l = 0; l < L; ++l) count += residues_[static_cast<size_t>(n * L + l)];
        for (int64_t l = 0; l < L; ++l) {
            const float m = residues_[static_cast<size_t>(n * L + l)];
            if (m == 0.0f) continue;
            for (int64_t j = 0; j < h; ++j) out[static_cast<size_t>((n * L + l) * h + j)] = static_cast<float>(m * g[static_cast<size_t>(n * h + j)] / count);
        }
    }
    return Tensor(last_shape_, backend_, out, backend_->device());
}

Tensor SequenceHead::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (last_shape_.rank() != 3) throw std::logic_error("SequenceHead::propagate_relevance: called before any forward()");
    if (pooling_ == Pooling::PerResidue) {
        const int64_t rows = last_shape_.dim(0) * last_shape_.dim(1);
        return Reshaped(linear_.propagate_relevance(Reshaped(relevance_out, Shape({rows, outputs_})), config), last_shape_);
    }
    const int64_t N = last_shape_.dim(0), L = last_shape_.dim(1), h = hidden_;
    const std::vector<float> rp = linear_.propagate_relevance(relevance_out, config).to_host_vector();  // (N, h)
    std::vector<float> out(static_cast<size_t>(N * L * h), 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        double count = 0;
        for (int64_t l = 0; l < L; ++l) count += residues_[static_cast<size_t>(n * L + l)];
        for (int64_t j = 0; j < h; ++j) {
            double pooled = 0;
            for (int64_t l = 0; l < L; ++l) pooled += residues_[static_cast<size_t>(n * L + l)] * last_input_[static_cast<size_t>((n * L + l) * h + j)] / count;
            const double stab = pooled + (pooled >= 0 ? config.epsilon : -config.epsilon);
            for (int64_t l = 0; l < L; ++l) {
                const double share = residues_[static_cast<size_t>(n * L + l)] * last_input_[static_cast<size_t>((n * L + l) * h + j)] / count;
                out[static_cast<size_t>((n * L + l) * h + j)] = static_cast<float>(share / stab * rp[static_cast<size_t>(n * h + j)]);
            }
        }
    }
    return Tensor(last_shape_, backend_, out, backend_->device());
}

std::vector<NamedParamRef> SequenceHead::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "linear", linear_);
    return out;
}

void SequenceHead::set_training(bool training) {
    Module::set_training(training);
    linear_.set_training(training);
}

void SequenceHead::release_activations() {
    linear_.release_activations();
    last_input_.clear();
    last_input_.shrink_to_fit();
}

Tensor ResidueMask(const Tensor& ids, const Tensor* keep, const TextTokenizer& tokenizer, DeviceBackend* backend) {
    if (ids.rank() != 2 || (keep != nullptr && keep->shape() != ids.shape())) {
        throw std::invalid_argument("ResidueMask: ids must be (N, L), and the padding mask the same shape");
    }
    const float cls = static_cast<float>(Id(tokenizer, "<cls>", "ResidueMask")), eos = static_cast<float>(Id(tokenizer, "<eos>", "ResidueMask"));
    std::vector<float> v = ids.to_host_vector(), k = keep != nullptr ? keep->to_host_vector() : std::vector<float>(v.size(), 1.0f);
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = k[i] != 0.0f && v[i] != cls && v[i] != eos ? 1.0f : 0.0f;
    return Tensor(ids.shape(), backend, out, backend->device());
}

}  // namespace pulsatrix
