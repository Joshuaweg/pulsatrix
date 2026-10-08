#include "pulsatrix/variant_scoring.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace pulsatrix {

std::vector<Mutation> ParseMutations(std::string_view text) {
    std::vector<Mutation> out;
    size_t begin = 0;
    while (begin <= text.size()) {
        size_t end = text.find(':', begin);
        if (end == std::string_view::npos) end = text.size();
        const std::string_view m = text.substr(begin, end - begin);
        bool ok = m.size() >= 3 && std::isalpha(static_cast<unsigned char>(m.front())) && std::isalpha(static_cast<unsigned char>(m.back()));
        for (size_t i = 1; ok && i + 1 < m.size(); ++i) ok = std::isdigit(static_cast<unsigned char>(m[i])) != 0;
        if (!ok) throw std::invalid_argument("ParseMutations: \"" + std::string(m) + "\" isn't like A23G");
        out.push_back({m.front(), std::stoll(std::string(m.substr(1, m.size() - 2))), m.back()});
        begin = end + 1;
    }
    return out;
}

namespace {

int64_t Index(std::string_view sequence, const Mutation& m, int64_t offset) {
    const int64_t i = m.position - offset;
    if (i < 0 || i >= static_cast<int64_t>(sequence.size())) {
        throw std::invalid_argument("mutation " + std::string(1, m.wild_type) + std::to_string(m.position) + std::string(1, m.mutant) +
                                    " is outside the sequence");
    }
    if (sequence[static_cast<size_t>(i)] != m.wild_type) {
        throw std::invalid_argument("mutation " + std::string(1, m.wild_type) + std::to_string(m.position) + std::string(1, m.mutant) +
                                    ": the sequence has " + std::string(1, sequence[static_cast<size_t>(i)]) + " there");
    }
    return i;
}

}  // namespace

std::string ApplyMutations(std::string_view sequence, const std::vector<Mutation>& mutations, int64_t offset) {
    std::string out(sequence);
    for (const Mutation& m : mutations) out[static_cast<size_t>(Index(sequence, m, offset))] = m.mutant;
    return out;
}

int64_t ScoringPassBytes(const EncoderLMConfig& config, int64_t length) {
    const int64_t H = config.num_attention_heads, d = config.hidden_size, L = length;
    // Scores, their scaled and masked copy, and the softmax module's input and output copies:
    // up to six (H, L, L) tensors at once. Q/K/V, context, residuals and LayerNorm copies are
    // about 16 (L, d); the MLP's two (L, d_ff); the logits and their log-softmax a few (L, V).
    const int64_t floats = 6 * H * L * L + 16 * L * d + 4 * L * config.intermediate_size + 3 * L * config.vocab_size;
    return floats * static_cast<int64_t>(sizeof(float));
}

std::pair<int64_t, int64_t> OptimalWindow(int64_t position, int64_t length, int64_t window) {
    const int64_t half = window / 2;
    if (length <= window) return {0, length};
    if (position < half) return {0, window};
    if (position >= length - half) return {length - window, length};
    return {std::max<int64_t>(0, position - half), std::min(length, position + half)};
}

VariantScorer::VariantScorer(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend, VariantScoringOptions options)
    : model_(model), tokenizer_(tokenizer), backend_(backend), options_(options) {
    const auto mask = tokenizer.token_to_id("<mask>");
    if (!mask) throw std::invalid_argument("VariantScorer: the tokenizer has no <mask>");
    mask_id_ = *mask;
    if (options_.window < 3 || options_.batch_size < 1 || options_.max_pass_bytes < 1) {
        throw std::invalid_argument("VariantScorer: window must be at least 3, batch_size and max_pass_bytes at least 1");
    }
}

int64_t VariantScorer::token_of(char residue) const {
    const auto id = tokenizer_.token_to_id(std::string(1, residue));
    if (!id) throw std::invalid_argument(std::string("VariantScorer: '") + residue + "' isn't a vocabulary token");
    return *id;
}

std::vector<int64_t> VariantScorer::Tokens(std::string_view sequence) const {
    if (sequence.empty()) throw std::invalid_argument("VariantScorer: the sequence is empty");
    std::vector<int64_t> ids{*tokenizer_.token_to_id("<cls>")};
    for (char c : sequence) ids.push_back(token_of(c));
    ids.push_back(*tokenizer_.token_to_id("<eos>"));
    return ids;
}

int64_t VariantScorer::BatchSize(int64_t length) const {
    const int64_t one = ScoringPassBytes(model_.config(), length);
    if (one > options_.max_pass_bytes) {
        throw std::invalid_argument("VariantScorer: one " + std::to_string(length) + "-token pass needs about " + std::to_string(one >> 20) +
                                    " MiB, over max_pass_bytes (" + std::to_string(options_.max_pass_bytes >> 20) +
                                    " MiB); raise it or shorten the window");
    }
    return std::min(options_.batch_size, options_.max_pass_bytes / one);
}

std::vector<float> VariantScorer::LogSoftmax(const std::vector<std::vector<int64_t>>& batch) {
    const auto N = static_cast<int64_t>(batch.size()), L = static_cast<int64_t>(batch.front().size());
    std::vector<float> ids;
    ids.reserve(static_cast<size_t>(N * L));
    for (const auto& row : batch) ids.insert(ids.end(), row.begin(), row.end());
    model_.clear_padding_mask();
    // Scoring needs only the logits: keep no activations, so a pass holds one layer's at a time.
    const bool keep = model_.keep_activations();
    model_.set_keep_activations(false);
    std::vector<float> logits;
    try {
        logits = model_.forward(Tensor(Shape({N, L}), backend_, ids, backend_->device())).to_host_vector();
    } catch (...) {
        model_.set_keep_activations(keep);
        throw;
    }
    model_.set_keep_activations(keep);
    const int64_t V = model_.config().vocab_size;
    for (int64_t r = 0; r < N * L; ++r) {
        float* row = logits.data() + r * V;
        const float m = *std::max_element(row, row + V);
        double z = 0;
        for (int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v] - m));
        const float log_z = m + static_cast<float>(std::log(z));
        for (int64_t v = 0; v < V; ++v) row[v] -= log_z;
    }
    return logits;
}

ResidueLogProbs VariantScorer::masked_marginals(std::string_view sequence) {
    const std::vector<int64_t> tokens = Tokens(sequence);
    const auto T = static_cast<int64_t>(tokens.size()), L = T - 2, V = model_.config().vocab_size;
    ResidueLogProbs out{L, V, std::vector<float>(static_cast<size_t>(L * V))};
    // Mask residue token i (1..L) inside its window; windows of one length batch together.
    std::map<int64_t, std::vector<int64_t>> by_length;
    for (int64_t i = 1; i <= L; ++i) {
        const auto [b, e] = OptimalWindow(i, T, options_.window);
        by_length[e - b].push_back(i);
    }
    for (const auto& [length, positions] : by_length) {
        const size_t batch_size = static_cast<size_t>(BatchSize(length));
        for (size_t first = 0; first < positions.size(); first += batch_size) {
            const size_t last = std::min(positions.size(), first + batch_size);
            std::vector<std::vector<int64_t>> batch;
            std::vector<int64_t> begins;
            for (size_t k = first; k < last; ++k) {
                const int64_t i = positions[k];
                const auto [b, e] = OptimalWindow(i, T, options_.window);
                std::vector<int64_t> masked(tokens.begin() + b, tokens.begin() + e);
                masked[static_cast<size_t>(i - b)] = mask_id_;
                batch.push_back(std::move(masked));
                begins.push_back(b);
            }
            const std::vector<float> lp = LogSoftmax(batch);
            for (size_t k = first; k < last; ++k) {
                const int64_t i = positions[k], n = static_cast<int64_t>(k - first), local = i - begins[static_cast<size_t>(n)];
                std::copy_n(lp.begin() + (n * length + local) * V, V, out.values.begin() + (i - 1) * V);
            }
        }
    }
    return out;
}

ResidueLogProbs VariantScorer::wild_type_marginals(std::string_view sequence) {
    // One pass over the whole sequence, as ProteinGym does (its "optimal" window setting doesn't
    // cut wild-type marginals).
    const std::vector<int64_t> tokens = Tokens(sequence);
    const auto L = static_cast<int64_t>(tokens.size()) - 2, V = model_.config().vocab_size;
    (void)BatchSize(L + 2);  // refuses a sequence too long for max_pass_bytes
    const std::vector<float> lp = LogSoftmax({tokens});
    return {L, V, std::vector<float>(lp.begin() + V, lp.begin() + (L + 1) * V)};
}

double VariantScorer::pseudo_log_likelihood(std::string_view sequence) {
    const ResidueLogProbs m = masked_marginals(sequence);
    double total = 0;
    for (int64_t i = 0; i < m.length; ++i) total += m.at(i, token_of(sequence[static_cast<size_t>(i)]));
    return total;
}

double VariantScorer::score(const ResidueLogProbs& marginals, std::string_view sequence, const std::vector<Mutation>& mutations,
                            int64_t offset) const {
    if (marginals.length != static_cast<int64_t>(sequence.size())) {
        throw std::invalid_argument("VariantScorer::score: the marginals weren't computed on this sequence");
    }
    double s = 0;
    for (const Mutation& m : mutations) {
        const int64_t i = Index(sequence, m, offset);
        s += static_cast<double>(marginals.at(i, token_of(m.mutant))) - marginals.at(i, token_of(m.wild_type));
    }
    return s;
}

std::vector<float> VariantScorer::single_mutant_scan(const ResidueLogProbs& marginals, std::string_view sequence) const {
    if (marginals.length != static_cast<int64_t>(sequence.size())) {
        throw std::invalid_argument("VariantScorer::single_mutant_scan: the marginals weren't computed on this sequence");
    }
    std::vector<float> scan(static_cast<size_t>(marginals.length * 20));
    for (int64_t i = 0; i < marginals.length; ++i) {
        const float wt = marginals.at(i, token_of(sequence[static_cast<size_t>(i)]));
        for (int64_t j = 0; j < 20; ++j) scan[static_cast<size_t>(i * 20 + j)] = marginals.at(i, token_of(kAminoAcids[j])) - wt;
    }
    return scan;
}

}  // namespace pulsatrix
