#include "pulsatrix/protein_explanations.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "portable_random.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explanation_metrics.hpp"
#include "pulsatrix/null_model_baseline.hpp"
#include "relevance_graph_detail.hpp"

namespace pulsatrix {

// ---- targets ---------------------------------------------------------------------------------

EncoderTarget EncoderTarget::MaskedToken(int64_t residue, char token) {
    EncoderTarget t;
    t.kind = Kind::MaskedToken;
    t.residue = residue;
    t.token = token;
    return t;
}

EncoderTarget EncoderTarget::ForMutation(const Mutation& m, int64_t offset) {
    EncoderTarget t;
    t.kind = Kind::Mutation;
    t.residue = m.position - offset;
    t.token = m.mutant;
    t.wild_type = m.wild_type;
    return t;
}

EncoderTarget EncoderTarget::ProteinHead(LinearHead head) {
    EncoderTarget t;
    t.kind = Kind::ProteinHead;
    t.head = std::move(head);
    return t;
}

EncoderTarget EncoderTarget::ResidueHead(LinearHead head, int64_t residue) {
    EncoderTarget t;
    t.kind = Kind::ResidueHead;
    t.head = std::move(head);
    t.residue = residue;
    return t;
}

std::string EncoderTarget::describe(std::string_view sequence) const {
    auto at = [&] {
        const char c = residue >= 0 && residue < static_cast<int64_t>(sequence.size()) ? sequence[static_cast<size_t>(residue)] : '?';
        return std::string(1, c) + std::to_string(residue + 1);
    };
    switch (kind) {
        case Kind::MaskedToken: return at() + " masked: " + std::string(1, token);
        case Kind::Mutation: return at() + std::string(1, token) + " log-odds";
        case Kind::ProteinHead: return "protein head";
        case Kind::ResidueHead: return "head at " + at();
    }
    return "";
}

double ResidueRelevance::total() const {
    return std::accumulate(residues.begin(), residues.end(), static_cast<double>(cls) + eos);
}

// ---- explainer -------------------------------------------------------------------------------

namespace {

bool Masks(const EncoderTarget& t) { return t.kind == EncoderTarget::Kind::MaskedToken || t.kind == EncoderTarget::Kind::Mutation; }

/** @brief The epsilon rule's stabilized denominator. */
double Stabilize(double z, double epsilon) { return z + (z >= 0 ? epsilon : -epsilon); }

}  // namespace

EncoderExplainer::EncoderExplainer(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend, LRPRuleConfig config)
    : model_(model), tokenizer_(tokenizer), backend_(backend), config_(config) {
    for (const char* t : {"<cls>", "<eos>", "<mask>"}) {
        if (!tokenizer.token_to_id(t)) throw std::invalid_argument(std::string("EncoderExplainer: the tokenizer has no ") + t);
    }
    mask_id_ = *tokenizer.token_to_id("<mask>");
}

std::vector<int64_t> EncoderExplainer::tokens(std::string_view sequence) const {
    if (sequence.empty()) throw std::invalid_argument("EncoderExplainer: the sequence is empty");
    std::vector<int64_t> ids{*tokenizer_.token_to_id("<cls>")};
    for (char c : sequence) {
        const auto id = tokenizer_.token_to_id(std::string(1, c));
        if (!id) throw std::invalid_argument(std::string("EncoderExplainer: '") + c + "' isn't a vocabulary token");
        ids.push_back(*id);
    }
    ids.push_back(*tokenizer_.token_to_id("<eos>"));
    return ids;
}

void EncoderExplainer::Check(std::string_view sequence, const EncoderTarget& t) const {
    const auto L = static_cast<int64_t>(sequence.size());
    const bool per_residue = t.kind != EncoderTarget::Kind::ProteinHead;
    if (per_residue && (t.residue < 0 || t.residue >= L)) {
        throw std::invalid_argument("EncoderExplainer: residue " + std::to_string(t.residue) + " is outside the sequence");
    }
    if (Masks(t) && !tokenizer_.token_to_id(std::string(1, t.token))) {
        throw std::invalid_argument(std::string("EncoderExplainer: '") + t.token + "' isn't a vocabulary token");
    }
    if (t.kind == EncoderTarget::Kind::Mutation && sequence[static_cast<size_t>(t.residue)] != t.wild_type) {
        throw std::invalid_argument("EncoderExplainer: the mutation's wild type " + std::string(1, t.wild_type) + " isn't the sequence's " +
                                    std::string(1, sequence[static_cast<size_t>(t.residue)]));
    }
    if (!Masks(t) && static_cast<int64_t>(t.head.weight.size()) != model_.config().hidden_size) {
        throw std::invalid_argument("EncoderExplainer: the head needs hidden_size weights");
    }
}

std::vector<float> EncoderExplainer::Forward(const std::vector<int64_t>& tokens, bool keep) {
    const auto T = static_cast<int64_t>(tokens.size());
    model_.clear_padding_mask();
    const bool was = model_.keep_activations();
    model_.set_keep_activations(keep);
    std::vector<float> logits;
    try {
        logits = model_.forward(Tensor(Shape({1, T}), backend_, std::vector<float>(tokens.begin(), tokens.end()), backend_->device()))
                     .to_host_vector();
    } catch (...) {
        model_.set_keep_activations(was);
        throw;
    }
    model_.set_keep_activations(was);
    return logits;
}

float EncoderExplainer::Value(const std::vector<float>& logits, const std::vector<float>& hidden, int64_t T, const EncoderTarget& t) const {
    const int64_t V = model_.config().vocab_size, h = model_.config().hidden_size;
    const int64_t pos = t.residue + 1;  // the token after <cls>
    auto logit = [&](char c) { return logits[static_cast<size_t>(pos * V + *tokenizer_.token_to_id(std::string(1, c)))]; };
    auto dot = [&](int64_t token) {
        double s = 0;
        for (int64_t j = 0; j < h; ++j) s += static_cast<double>(hidden[static_cast<size_t>(token * h + j)]) * t.head.weight[static_cast<size_t>(j)];
        return s;
    };
    switch (t.kind) {
        case EncoderTarget::Kind::MaskedToken: return logit(t.token);
        case EncoderTarget::Kind::Mutation: return logit(t.token) - logit(t.wild_type);
        case EncoderTarget::Kind::ProteinHead: {
            double s = 0;
            for (int64_t i = 1; i + 1 < T; ++i) s += dot(i);
            return static_cast<float>(s / static_cast<double>(T - 2) + t.head.bias);
        }
        case EncoderTarget::Kind::ResidueHead: return static_cast<float>(dot(pos) + t.head.bias);
    }
    return 0.0f;
}

float EncoderExplainer::evaluate(const std::vector<int64_t>& tokens, const EncoderTarget& target) {
    std::vector<int64_t> ids = tokens;
    if (Masks(target)) ids.at(static_cast<size_t>(target.residue + 1)) = mask_id_;
    const std::vector<float> logits = Forward(ids, false);
    return Value(logits, Masks(target) ? std::vector<float>() : model_.last_hidden_state().to_host_vector(), static_cast<int64_t>(ids.size()),
                 target);
}

ResidueRelevance EncoderExplainer::explain(std::string_view sequence, const EncoderTarget& target) {
    return Explain(sequence, target, false, nullptr, nullptr);
}

ResidueRelevance EncoderExplainer::Explain(std::string_view sequence, const EncoderTarget& target, bool keep,
                                           std::vector<std::vector<float>>* boundaries_out, std::vector<float>* logits_out) {
    Check(sequence, target);
    std::vector<int64_t> ids = tokens(sequence);
    const auto T = static_cast<int64_t>(ids.size());
    const int64_t V = model_.config().vocab_size, h = model_.config().hidden_size, pos = target.residue + 1;
    if (Masks(target)) ids[static_cast<size_t>(pos)] = mask_id_;
    const std::vector<float> logits = Forward(ids, true);
    const std::vector<float> hidden = model_.last_hidden_state().to_host_vector();
    const float value = Value(logits, hidden, T, target);

    std::vector<Tensor> boundaries;
    if (Masks(target)) {
        // Relevance at the logits is the explained value's terms, as value.backward() seeds it.
        std::vector<float> seed(static_cast<size_t>(T * V), 0.0f);
        const int64_t tok = *tokenizer_.token_to_id(std::string(1, target.token));
        seed[static_cast<size_t>(pos * V + tok)] = logits[static_cast<size_t>(pos * V + tok)];
        if (target.kind == EncoderTarget::Kind::Mutation) {
            const int64_t wild = *tokenizer_.token_to_id(std::string(1, target.wild_type));
            seed[static_cast<size_t>(pos * V + wild)] = -logits[static_cast<size_t>(pos * V + wild)];
        }
        boundaries = model_.propagate_relevance_by_layer(Tensor(Shape({1, T, V}), backend_, seed, backend_->device()), config_);
    } else {
        // A linear head: each representation element's share of the output, by the epsilon rule
        // with the bias in the denominator (LXT's gradient * input).
        std::vector<float> seed(static_cast<size_t>(T * h), 0.0f);
        const double scale = value / Stabilize(value, config_.epsilon);
        const bool pooled = target.kind == EncoderTarget::Kind::ProteinHead;
        for (int64_t i = pooled ? 1 : pos; i <= (pooled ? T - 2 : pos); ++i) {
            for (int64_t j = 0; j < h; ++j) {
                const size_t k = static_cast<size_t>(i * h + j);
                const double contribution = static_cast<double>(hidden[k]) * target.head.weight[static_cast<size_t>(j)] / (pooled ? T - 2 : 1);
                seed[k] = static_cast<float>(contribution * scale);
            }
        }
        boundaries = model_.propagate_hidden_relevance_by_layer(Tensor(Shape({1, T, h}), backend_, seed, backend_->device()), config_);
    }

    ResidueRelevance out;
    out.sequence = std::string(sequence);
    out.target = target.describe(sequence);
    out.value = value;
    for (const Tensor& b : boundaries) {
        const std::vector<float> r = b.to_host_vector();
        if (boundaries_out != nullptr) boundaries_out->push_back(r);
        std::vector<float> per_token(static_cast<size_t>(T), 0.0f);
        for (int64_t t = 0; t < T; ++t) {
            double s = 0;
            for (int64_t j = 0; j < h; ++j) s += r[static_cast<size_t>(t * h + j)];
            per_token[static_cast<size_t>(t)] = static_cast<float>(s);
        }
        out.layers.push_back(std::move(per_token));
    }
    // The token-dropout scale is a constant, so the embeddings' relevance is the tokens'.
    const std::vector<float>& first = out.layers.front();
    out.cls = first.front();
    out.eos = first.back();
    out.residues.assign(first.begin() + 1, first.end() - 1);
    if (!keep) model_.release_activations();
    if (logits_out != nullptr) *logits_out = logits;
    return out;
}

AttributionGraph EncoderExplainer::relevance_graph(std::string_view sequence, const EncoderTarget& target, const RelevanceGraphOptions& options) {
    const auto T = static_cast<int64_t>(sequence.size()) + 2;
    if (T > options.max_tokens) {
        throw std::invalid_argument("EncoderExplainer::relevance_graph: " + std::to_string(T) + " tokens, more than max_tokens (" +
                                    std::to_string(options.max_tokens) + ")");
    }
    relevance_graph_detail::GraphInputs in;
    std::vector<float> logits;
    const ResidueRelevance r = Explain(sequence, target, true, &in.boundaries, &logits);
    const int64_t h = model_.config().hidden_size, V = model_.config().vocab_size;
    in.positions = T;
    in.hidden = h;
    in.layers = model_.num_layers();
    in.ids = tokens(sequence);
    if (Masks(target)) in.ids[static_cast<size_t>(target.residue + 1)] = mask_id_;
    in.block_relevance = [&](int64_t block, const std::vector<float>& rel) {
        return model_.layer(block)
            .propagate_relevance(Tensor(Shape({1, T, h}), backend_, rel, backend_->device()), config_)
            .to_host_vector();
    };
    in.output_position = target.kind == EncoderTarget::Kind::ProteinHead ? 0 : target.residue + 1;
    in.output_value = r.value;
    in.output_label = r.target;
    if (Masks(target)) {
        const int64_t tok = *tokenizer_.token_to_id(std::string(1, target.token));
        const float* row = logits.data() + in.output_position * V;
        const float top = *std::max_element(row, row + V);
        double z = 0;
        for (int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v] - top));
        in.output_feature = tok;
        in.output_probability = std::exp(static_cast<double>(row[tok] - top)) / z;
    }
    in.description = "AttnLRP relevance for " + r.target + " through the residual stream: one node per layer and residue";
    AttributionGraph g;
    try {
        g = relevance_graph_detail::Assemble(in, options);
    } catch (...) {
        model_.release_activations();
        throw;
    }
    model_.release_activations();
    return g;
}

// ---- checks ----------------------------------------------------------------------------------

ResidueAgreement CompareResidueSignals(const std::vector<float>& relevance, const std::vector<float>& signal, double top_fraction) {
    if (relevance.size() != signal.size()) throw std::invalid_argument("CompareResidueSignals: the lengths differ");
    if (!(top_fraction > 0 && top_fraction <= 1)) throw std::invalid_argument("CompareResidueSignals: top_fraction must be in (0, 1]");
    std::vector<float> r, s;
    for (size_t i = 0; i < relevance.size(); ++i) {
        if (std::isfinite(relevance[i]) && std::isfinite(signal[i])) {
            r.push_back(std::abs(relevance[i]));
            s.push_back(signal[i]);
        }
    }
    if (r.size() < 3) throw std::invalid_argument("CompareResidueSignals: fewer than 3 residues have both values");
    ResidueAgreement a;
    a.compared = static_cast<int64_t>(r.size());
    a.spearman = SpearmanRankCorrelation(r, s);
    a.top = std::max<int64_t>(1, static_cast<int64_t>(std::floor(top_fraction * static_cast<double>(r.size()))));
    auto top = [&](const std::vector<float>& v) {
        std::vector<size_t> order(v.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return v[x] > v[y]; });
        order.resize(static_cast<size_t>(a.top));
        std::sort(order.begin(), order.end());
        return order;
    };
    const std::vector<size_t> tr = top(r), ts = top(s);
    std::vector<size_t> both;
    std::set_intersection(tr.begin(), tr.end(), ts.begin(), ts.end(), std::back_inserter(both));
    a.top_overlap = static_cast<double>(both.size()) / static_cast<double>(a.top);
    a.chance_overlap = static_cast<double>(a.top) / static_cast<double>(a.compared);
    return a;
}

std::vector<float> DmsPositionSensitivity(const DmsVariants& variants, std::string_view sequence, int64_t offset) {
    std::vector<double> sum(sequence.size(), 0.0);
    std::vector<int64_t> count(sequence.size(), 0);
    for (size_t v = 0; v < variants.mutants.size(); ++v) {
        const std::vector<Mutation> m = ParseMutations(variants.mutants[v]);
        if (m.size() != 1) continue;
        (void)ApplyMutations(sequence, m, offset);  // checks the wild type
        const auto i = static_cast<size_t>(m[0].position - offset);
        sum[i] += variants.scores[v];
        ++count[i];
    }
    std::vector<float> out(sequence.size(), std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < out.size(); ++i) {
        if (count[i] > 0) out[i] = static_cast<float>(-sum[i] / static_cast<double>(count[i]));
    }
    return out;
}

namespace {

/** @brief Each record's match columns, and, for the query, which of its residues each column is. */
struct MatchColumns {
    std::vector<std::string> rows;      // one string of match-column characters per record
    std::vector<int64_t> query_residue; // per column: the query residue it holds, or -1
    int64_t query_length = 0;
};

MatchColumns Columns(const std::vector<FastaRecord>& alignment) {
    if (alignment.empty()) throw std::invalid_argument("AlignmentConservation: the alignment is empty");
    MatchColumns m;
    const std::string& query = alignment.front().sequence;
    bool aligned = true;  // ProteinGym's layout: every row as long as the query
    for (const FastaRecord& r : alignment) aligned = aligned && r.sequence.size() == query.size();
    for (const FastaRecord& r : alignment) {
        std::string row;
        for (char c : r.sequence) {
            const bool match = aligned || std::isupper(static_cast<unsigned char>(c)) || c == '-';
            if (match) row.push_back(c);
        }
        m.rows.push_back(std::move(row));
    }
    const size_t width = m.rows.front().size();
    for (const std::string& row : m.rows) {
        if (row.size() != width) throw std::invalid_argument("AlignmentConservation: the records' match columns don't line up");
    }
    // The query's residues are its letters; a column holds one when the query has an uppercase
    // letter there (a focus column).
    int64_t residue = 0;
    size_t column = 0;
    for (char c : query) {
        const bool letter = std::isalpha(static_cast<unsigned char>(c)) != 0;
        const bool in_columns = aligned || std::isupper(static_cast<unsigned char>(c)) || c == '-';
        if (in_columns) {
            m.query_residue.push_back(letter && std::isupper(static_cast<unsigned char>(c)) ? residue : -1);
            ++column;
        }
        if (letter) ++residue;
    }
    m.query_length = residue;
    return m;
}

}  // namespace

std::string AlignmentQuery(const std::vector<FastaRecord>& alignment) {
    if (alignment.empty()) throw std::invalid_argument("AlignmentQuery: the alignment is empty");
    std::string q;
    for (char c : alignment.front().sequence) {
        if (std::isalpha(static_cast<unsigned char>(c))) q.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return q;
}

std::vector<float> AlignmentConservation(const std::vector<FastaRecord>& alignment) {
    const MatchColumns m = Columns(alignment);
    static const std::string kAa = kAminoAcids;
    std::vector<float> out(static_cast<size_t>(m.query_length), std::numeric_limits<float>::quiet_NaN());
    const double max_bits = std::log2(20.0);
    for (size_t col = 0; col < m.query_residue.size(); ++col) {
        const int64_t residue = m.query_residue[col];
        if (residue < 0) continue;
        int64_t counts[20] = {};
        int64_t residues = 0;
        for (const std::string& row : m.rows) {
            const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(row[col])));
            const size_t a = kAa.find(c);
            if (a == std::string::npos) continue;  // a gap, or a rare or ambiguous letter
            ++counts[a];
            ++residues;
        }
        if (residues == 0) {
            out[static_cast<size_t>(residue)] = 0.0f;
            continue;
        }
        double entropy = 0;
        for (int64_t c : counts) {
            if (c == 0) continue;
            const double p = static_cast<double>(c) / static_cast<double>(residues);
            entropy -= p * std::log2(p);
        }
        const double coverage = static_cast<double>(residues) / static_cast<double>(m.rows.size());
        out[static_cast<size_t>(residue)] = static_cast<float>((max_bits - entropy) * coverage);
    }
    return out;
}

std::vector<float> RelevanceProfile(EncoderExplainer& explainer, std::string_view sequence,
                                    const std::function<void(int64_t, int64_t)>& progress) {
    const auto L = static_cast<int64_t>(sequence.size());
    std::vector<double> sum(static_cast<size_t>(L), 0.0);
    for (int64_t t = 0; t < L; ++t) {
        const ResidueRelevance r = explainer.explain(sequence, EncoderTarget::MaskedToken(t, sequence[static_cast<size_t>(t)]));
        double total = 0;
        for (float v : r.residues) total += std::abs(static_cast<double>(v));
        if (total > 0) {
            for (int64_t i = 0; i < L; ++i) sum[static_cast<size_t>(i)] += std::abs(static_cast<double>(r.residues[static_cast<size_t>(i)])) / total;
        }
        if (progress) progress(t + 1, L);
    }
    std::vector<float> out(static_cast<size_t>(L));
    for (int64_t i = 0; i < L; ++i) out[static_cast<size_t>(i)] = static_cast<float>(sum[static_cast<size_t>(i)] / static_cast<double>(L));
    return out;
}

ResidueDeletionCurve DeletionCheck(EncoderExplainer& explainer, std::string_view sequence, const EncoderTarget& target,
                                   const std::vector<float>& relevance, int64_t steps, int64_t random_orders, uint64_t seed) {
    if (relevance.size() != sequence.size()) throw std::invalid_argument("DeletionCheck: needs one relevance value per residue");
    if (steps < 1 || random_orders < 1) throw std::invalid_argument("DeletionCheck: steps and random_orders must be at least 1");
    const std::vector<int64_t> base = explainer.tokens(sequence);
    const bool masks = target.kind == EncoderTarget::Kind::MaskedToken || target.kind == EncoderTarget::Kind::Mutation;
    std::vector<size_t> residues;
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (!(masks && static_cast<int64_t>(i) == target.residue)) residues.push_back(i);
    }
    auto curve = [&](const std::vector<size_t>& order) {
        std::vector<float> scores;
        for (int64_t s = 0; s <= steps; ++s) {
            const auto n = static_cast<size_t>(std::llround(static_cast<double>(s) / static_cast<double>(steps) * static_cast<double>(order.size())));
            std::vector<int64_t> ids = base;
            for (size_t k = 0; k < n; ++k) ids[order[k] + 1] = explainer.mask_id();
            scores.push_back(explainer.evaluate(ids, target));
        }
        return scores;
    };
    auto area = [&](const std::vector<float>& y) {
        double a = 0;
        for (size_t s = 1; s < y.size(); ++s) a += (static_cast<double>(y[s - 1]) + y[s]) / 2.0 / static_cast<double>(steps);
        return a;
    };
    ResidueDeletionCurve out;
    for (int64_t s = 0; s <= steps; ++s) out.fractions.push_back(static_cast<float>(static_cast<double>(s) / static_cast<double>(steps)));
    // Residues that support the value come first: relevance of the value's own sign, largest first.
    const float sign = explainer.evaluate(base, target) < 0 ? -1.0f : 1.0f;
    out.sign = sign;
    std::vector<size_t> order = residues;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sign * relevance[a] > sign * relevance[b]; });
    out.scores = curve(order);
    out.random_scores.assign(static_cast<size_t>(steps + 1), 0.0f);
    PortableRng rng{seed};
    for (int64_t r = 0; r < random_orders; ++r) {
        std::vector<size_t> shuffled = residues;
        for (size_t i = shuffled.size(); i > 1; --i) std::swap(shuffled[i - 1], shuffled[static_cast<size_t>(rng.next() % i)]);
        const std::vector<float> s = curve(shuffled);
        for (size_t k = 0; k < s.size(); ++k) out.random_scores[k] += s[k] / static_cast<float>(random_orders);
    }
    std::vector<float> toward(out.scores), random_toward(out.random_scores);
    for (float& v : toward) v *= sign;
    for (float& v : random_toward) v *= sign;
    out.auc = area(toward);
    out.random_auc = area(random_toward);
    return out;
}

RandomizationCheckResult RandomizationCheck(EncoderExplainer& explainer, std::string_view sequence, const EncoderTarget& target, uint64_t seed) {
    EncoderLM& model = explainer.model();
    std::vector<std::string> layers = {"lm_head"};
    for (int64_t i = model.num_layers(); i-- > 0;) layers.push_back("layers." + std::to_string(i));
    layers.push_back("embed_tokens");
    auto magnitudes = [&] {
        std::vector<float> m = explainer.explain(sequence, target).residues;
        for (float& v : m) v = std::abs(v);
        return m;
    };
    ParameterSnapshot saved(model);  // restores the model however this returns
    const std::vector<float> original = magnitudes();
    RandomizationCheckResult out;
    PortableRng seeds{seed};
    for (const std::string& layer : layers) {
        // Weight matrices only: redrawing a norm's gain from its own small spread would shrink it
        // toward zero and leave the block a bare residual connection, hiding the randomization.
        for (const NamedParamRef& p : model.named_parameters()) {
            const bool in_layer = p.name == layer || p.name.rfind(layer + ".", 0) == 0;
            if (in_layer && p.ref.value->rank() >= 2) ReinitializeParameters(model, seeds.next(), p.name);
        }
        out.layers.push_back(layer);
        out.similarity.push_back(SpearmanRankCorrelation(original, magnitudes()));
    }
    return out;
}

}  // namespace pulsatrix
