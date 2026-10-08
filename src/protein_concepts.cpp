#include "pulsatrix/protein_concepts.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "portable_random.hpp"
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/fitness_metrics.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {

// ---- annotations -----------------------------------------------------------------------------

std::vector<AnnotatedProtein> ParseAnnotatedProteins(std::string_view jsonl) {
    std::vector<AnnotatedProtein> out;
    size_t pos = 0, line_number = 0;
    while (pos < jsonl.size()) {
        const size_t eol = std::min(jsonl.find('\n', pos), jsonl.size());
        const std::string_view line = jsonl.substr(pos, eol - pos);
        pos = eol + 1;
        ++line_number;
        if (line.find_first_not_of(" \t\r") == std::string_view::npos) continue;
        const std::string where = "ParseAnnotatedProteins: line " + std::to_string(line_number);
        JsonValue v;
        try {
            v = ParseJson(line);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(where + ": " + e.what());
        }
        const JsonValue* acc = v.find("accession");
        const JsonValue* seq = v.find("sequence");
        const JsonValue* features = v.find("features");
        if (acc == nullptr || seq == nullptr || features == nullptr || acc->type() != JsonValue::Type::String ||
            seq->type() != JsonValue::Type::String || features->type() != JsonValue::Type::Array) {
            throw std::invalid_argument(where + ": needs a string accession and sequence and a features array");
        }
        AnnotatedProtein p{acc->as_string(), seq->as_string(), {}};
        const auto L = static_cast<int64_t>(p.sequence.size());
        for (const JsonValue& f : features->as_array()) {
            const JsonValue* type = f.find("type");
            const JsonValue* start = f.find("start");
            const JsonValue* end = f.find("end");
            if (type == nullptr || start == nullptr || end == nullptr || type->type() != JsonValue::Type::String) {
                throw std::invalid_argument(where + ": a feature needs type, start and end");
            }
            ProteinFeature x{type->as_string(), start->as_int64(), end->as_int64()};
            if (x.start < 1 || x.end < x.start || x.end > L) throw std::invalid_argument(where + ": a feature is outside the sequence");
            p.features.push_back(std::move(x));
        }
        out.push_back(std::move(p));
    }
    return out;
}

std::vector<AnnotatedProtein> ReadAnnotatedProteins(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ReadAnnotatedProteins: can't read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return ParseAnnotatedProteins(text.str());
}

std::vector<int> ConceptLabels(const AnnotatedProtein& protein, const std::string& type) {
    std::vector<int> out(protein.sequence.size(), 0);
    for (const ProteinFeature& f : protein.features) {
        if (f.type != type) continue;
        if (type == "Disulfide bond") {
            out[static_cast<size_t>(f.start - 1)] = 1;
            out[static_cast<size_t>(f.end - 1)] = 1;
            continue;
        }
        for (int64_t i = f.start; i <= f.end; ++i) out[static_cast<size_t>(i - 1)] = 1;
    }
    return out;
}

std::vector<int> SecondaryStructureLabels(const AnnotatedProtein& protein) {
    std::vector<int> out(protein.sequence.size(), 2);
    for (const ProteinFeature& f : protein.features) {
        const int c = f.type == "Helix" ? 0 : (f.type == "Beta strand" ? 1 : -1);
        if (c < 0) continue;
        for (int64_t i = f.start; i <= f.end; ++i) out[static_cast<size_t>(i - 1)] = c;
    }
    return out;
}

std::vector<std::pair<std::string, int64_t>> ConceptCounts(const std::vector<AnnotatedProtein>& proteins) {
    std::map<std::string, int64_t> counts;
    for (const AnnotatedProtein& p : proteins) {
        std::map<std::string, std::vector<int>> seen;
        for (const ProteinFeature& f : p.features) {
            if (!seen.count(f.type)) seen[f.type] = ConceptLabels(p, f.type);
        }
        for (const auto& [type, labels] : seen) counts[type] += std::accumulate(labels.begin(), labels.end(), int64_t{0});
    }
    std::vector<std::pair<std::string, int64_t>> out(counts.begin(), counts.end());
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return out;
}

// ---- representations ---------------------------------------------------------------------------

ResidueEmbeddings EmbedResidues(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend,
                                const std::vector<std::string>& sequences, const std::vector<int64_t>& layers, int64_t max_tokens) {
    const int64_t n_layers = model.num_layers(), h = model.config().hidden_size;
    for (int64_t l : layers) {
        if (l < 0 || l > n_layers) throw std::invalid_argument("EmbedResidues: layer " + std::to_string(l) + " is outside [0, num_layers]");
    }
    const auto cls = tokenizer.token_to_id("<cls>"), eos = tokenizer.token_to_id("<eos>");
    if (!cls || !eos) throw std::invalid_argument("EmbedResidues: the tokenizer lacks <cls> or <eos>");
    ResidueEmbeddings out;
    out.layers = layers;
    out.hidden = h;
    out.values.resize(layers.size());
    const bool keep = model.keep_activations();
    model.set_keep_activations(true);
    model.clear_padding_mask();
    try {
        for (const std::string& s : sequences) {
            if (static_cast<int64_t>(s.size()) + 2 > max_tokens) throw std::invalid_argument("EmbedResidues: a sequence is longer than max_tokens - 2");
            std::vector<float> ids{static_cast<float>(*cls)};
            for (char c : s) {
                const auto id = tokenizer.token_to_id(std::string(1, c));
                if (!id) throw std::invalid_argument(std::string("EmbedResidues: '") + c + "' isn't a vocabulary token");
                ids.push_back(static_cast<float>(*id));
            }
            ids.push_back(static_cast<float>(*eos));
            const auto T = static_cast<int64_t>(ids.size());
            (void)model.forward(Tensor(Shape({1, T}), backend, ids, backend->device()));
            out.offsets.push_back(out.residues);
            for (size_t k = 0; k < layers.size(); ++k) {
                const std::vector<float> all =
                    (layers[k] == n_layers ? model.last_hidden_state() : model.hidden_states()[static_cast<size_t>(layers[k])]).to_host_vector();
                out.values[k].insert(out.values[k].end(), all.begin() + h, all.end() - h);  // drop <cls> and <eos>
            }
            out.residues += T - 2;
            model.release_activations();
        }
    } catch (...) {
        model.set_keep_activations(keep);
        throw;
    }
    model.set_keep_activations(keep);
    return out;
}

// ---- probes ------------------------------------------------------------------------------------

std::vector<float> SequenceWindowFeatures(const std::vector<std::string>& sequences, int64_t window) {
    if (window < 0) throw std::invalid_argument("SequenceWindowFeatures: window must not be negative");
    static const std::string kAa = "ACDEFGHIKLMNPQRSTVWY";
    const int64_t width = 20 * (2 * window + 1);
    std::vector<float> out;
    for (const std::string& s : sequences) {
        const auto L = static_cast<int64_t>(s.size());
        const size_t base = out.size();
        out.resize(base + static_cast<size_t>(L * width), 0.0f);
        for (int64_t i = 0; i < L; ++i) {
            for (int64_t k = -window; k <= window; ++k) {
                if (i + k < 0 || i + k >= L) continue;
                const size_t a = kAa.find(s[static_cast<size_t>(i + k)]);
                if (a == std::string::npos) continue;
                out[base + static_cast<size_t>(i * width + (k + window) * 20) + a] = 1.0f;
            }
        }
    }
    return out;
}

ProbeResult TrainLinearProbe(const std::vector<float>& features, int64_t dim, const std::vector<int>& labels, int classes,
                             const std::vector<bool>& train, const std::vector<bool>& test, DeviceBackend* backend, const ProbeOptions& o) {
    const auto N = static_cast<int64_t>(labels.size());
    if (dim < 1 || features.size() != static_cast<size_t>(N * dim) || train.size() != labels.size() || test.size() != labels.size() || classes < 2) {
        throw std::invalid_argument("TrainLinearProbe: features must be residues x dim with one label and flag per residue, and classes >= 2");
    }
    std::vector<int64_t> train_rows, test_rows;
    for (int64_t i = 0; i < N; ++i) {
        if (labels[static_cast<size_t>(i)] < 0 || labels[static_cast<size_t>(i)] >= classes) throw std::invalid_argument("TrainLinearProbe: a label is outside [0, classes)");
        if (train[static_cast<size_t>(i)]) train_rows.push_back(i);
        if (test[static_cast<size_t>(i)]) test_rows.push_back(i);
    }
    if (train_rows.empty() || test_rows.empty()) throw std::invalid_argument("TrainLinearProbe: the train and test sets must not be empty");
    // Standardize by the training residues.
    std::vector<double> mean(static_cast<size_t>(dim), 0.0), sd(static_cast<size_t>(dim), 0.0);
    for (int64_t r : train_rows) {
        for (int64_t j = 0; j < dim; ++j) mean[static_cast<size_t>(j)] += features[static_cast<size_t>(r * dim + j)];
    }
    for (double& m : mean) m /= static_cast<double>(train_rows.size());
    for (int64_t r : train_rows) {
        for (int64_t j = 0; j < dim; ++j) {
            const double d = features[static_cast<size_t>(r * dim + j)] - mean[static_cast<size_t>(j)];
            sd[static_cast<size_t>(j)] += d * d;
        }
    }
    for (double& s : sd) s = std::sqrt(s / static_cast<double>(train_rows.size())) + 1e-6;
    auto batch_of = [&](const std::vector<int64_t>& rows, size_t from, size_t to, std::vector<float>& x, std::vector<float>& y) {
        x.resize((to - from) * static_cast<size_t>(dim));
        y.resize(to - from);
        for (size_t k = from; k < to; ++k) {
            const int64_t r = rows[k];
            for (int64_t j = 0; j < dim; ++j) {
                x[(k - from) * static_cast<size_t>(dim) + static_cast<size_t>(j)] =
                    static_cast<float>((features[static_cast<size_t>(r * dim + j)] - mean[static_cast<size_t>(j)]) / sd[static_cast<size_t>(j)]);
            }
            y[k - from] = static_cast<float>(labels[static_cast<size_t>(r)]);
        }
    };
    LinearModule probe(dim, classes, backend);
    AdamWOptimizer opt(o.learning_rate, backend, o.weight_decay);
    TokenCrossEntropyLoss loss(backend);
    PortableRng rng{o.seed};
    std::vector<float> x, y;
    for (int64_t epoch = 0; epoch < o.epochs; ++epoch) {
        for (size_t i = train_rows.size(); i > 1; --i) std::swap(train_rows[i - 1], train_rows[static_cast<size_t>(rng.next() % i)]);
        for (size_t from = 0; from < train_rows.size(); from += static_cast<size_t>(o.batch)) {
            const size_t to = std::min(train_rows.size(), from + static_cast<size_t>(o.batch));
            batch_of(train_rows, from, to, x, y);
            const auto B = static_cast<int64_t>(to - from);
            opt.zero_grad(probe);
            const Tensor logits = probe.forward(Tensor(Shape({B, dim}), backend, x, backend->device()));
            (void)loss.forward(logits, Tensor(Shape({B}), backend, y, backend->device()));
            (void)probe.backward(loss.backward());
            opt.step(probe);
        }
    }
    // Score the held-out residues.
    std::vector<int64_t> correct(static_cast<size_t>(classes), 0), count(static_cast<size_t>(classes), 0);
    std::vector<int> binary_labels;
    std::vector<double> binary_scores;
    for (size_t from = 0; from < test_rows.size(); from += static_cast<size_t>(o.batch)) {
        const size_t to = std::min(test_rows.size(), from + static_cast<size_t>(o.batch));
        batch_of(test_rows, from, to, x, y);
        const auto B = static_cast<int64_t>(to - from);
        const std::vector<float> logits = probe.forward(Tensor(Shape({B, dim}), backend, x, backend->device())).to_host_vector();
        for (int64_t b = 0; b < B; ++b) {
            const float* row = logits.data() + b * classes;
            const int pred = static_cast<int>(std::max_element(row, row + classes) - row);
            const int truth = static_cast<int>(y[static_cast<size_t>(b)]);
            ++count[static_cast<size_t>(truth)];
            correct[static_cast<size_t>(truth)] += pred == truth ? 1 : 0;
            if (classes == 2) {
                binary_labels.push_back(truth);
                binary_scores.push_back(static_cast<double>(row[1]) - row[0]);  // the log-odds of class 1
            }
        }
    }
    ProbeResult r;
    r.train = static_cast<int64_t>(train_rows.size());
    r.test = static_cast<int64_t>(test_rows.size());
    r.accuracy = static_cast<double>(std::accumulate(correct.begin(), correct.end(), int64_t{0})) / static_cast<double>(r.test);
    double recall_sum = 0;
    int present = 0;
    for (int c = 0; c < classes; ++c) {
        if (count[static_cast<size_t>(c)] == 0) continue;
        recall_sum += static_cast<double>(correct[static_cast<size_t>(c)]) / static_cast<double>(count[static_cast<size_t>(c)]);
        ++present;
    }
    r.balanced_accuracy = present > 0 ? recall_sum / present : 0.0;
    r.auc = classes == 2 ? RocAuc(binary_labels, binary_scores) : std::numeric_limits<double>::quiet_NaN();
    return r;
}

// ---- features ----------------------------------------------------------------------------------

SparseCodes EncodeSparse(Featurizer& featurizer, const std::vector<float>& rows, DeviceBackend* backend, int64_t batch) {
    const int64_t d = featurizer.input_dim(), m = featurizer.num_features();
    if (rows.size() % static_cast<size_t>(d) != 0) throw std::invalid_argument("EncodeSparse: rows must be whole rows of input_dim");
    const auto N = static_cast<int64_t>(rows.size() / static_cast<size_t>(d));
    SparseCodes c;
    c.num_features = m;
    c.max_value.assign(static_cast<size_t>(m), 0.0f);
    c.row_start.push_back(0);
    for (int64_t from = 0; from < N; from += batch) {
        const int64_t B = std::min(batch, N - from);
        const std::vector<float> x(rows.begin() + from * d, rows.begin() + (from + B) * d);
        const std::vector<float> f = featurizer.encode(Tensor(Shape({B, d}), backend, x, backend->device())).to_host_vector();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t j = 0; j < m; ++j) {
                const float v = f[static_cast<size_t>(b * m + j)];
                if (v <= 0.0f) continue;
                c.feature.push_back(static_cast<int32_t>(j));
                c.value.push_back(v);
                c.max_value[static_cast<size_t>(j)] = std::max(c.max_value[static_cast<size_t>(j)], v);
            }
            c.row_start.push_back(static_cast<int64_t>(c.feature.size()));
        }
    }
    return c;
}

SparseCodes NeuronCodes(const std::vector<float>& rows, int64_t dim) {
    if (dim < 1 || rows.size() % static_cast<size_t>(dim) != 0) throw std::invalid_argument("NeuronCodes: rows must be whole rows of dim");
    SparseCodes c;
    c.num_features = 2 * dim;
    c.max_value.assign(static_cast<size_t>(2 * dim), 0.0f);
    c.row_start.push_back(0);
    for (size_t r = 0; r < rows.size() / static_cast<size_t>(dim); ++r) {
        for (int64_t j = 0; j < dim; ++j) {
            const float v = rows[r * static_cast<size_t>(dim) + static_cast<size_t>(j)];
            if (v == 0.0f) continue;
            const int32_t f = static_cast<int32_t>(v > 0 ? 2 * j : 2 * j + 1);
            c.feature.push_back(f);
            c.value.push_back(std::abs(v));
            c.max_value[static_cast<size_t>(f)] = std::max(c.max_value[static_cast<size_t>(f)], std::abs(v));
        }
        c.row_start.push_back(static_cast<int64_t>(c.feature.size()));
    }
    return c;
}

ConceptMatch MatchConcept(const SparseCodes& codes, const std::vector<int>& labels, const std::string& concept_name,
                          const std::vector<double>& thresholds) {
    if (static_cast<int64_t>(labels.size()) != codes.residues()) throw std::invalid_argument("MatchConcept: needs one label per residue");
    const size_t m = static_cast<size_t>(codes.num_features), k = thresholds.size();
    std::vector<int64_t> predicted(m * k, 0), hits(m * k, 0);
    ConceptMatch out;
    out.concept_name = concept_name;
    for (int64_t r = 0; r < codes.residues(); ++r) {
        const bool positive = labels[static_cast<size_t>(r)] != 0;
        out.positives += positive ? 1 : 0;
        for (int64_t e = codes.row_start[static_cast<size_t>(r)]; e < codes.row_start[static_cast<size_t>(r) + 1]; ++e) {
            const auto f = static_cast<size_t>(codes.feature[static_cast<size_t>(e)]);
            const double v = codes.value[static_cast<size_t>(e)];
            for (size_t t = 0; t < k; ++t) {
                if (v <= thresholds[t] * codes.max_value[f]) continue;
                ++predicted[f * k + t];
                hits[f * k + t] += positive ? 1 : 0;
            }
        }
    }
    for (size_t f = 0; f < m; ++f) {
        double best = 0;
        for (size_t t = 0; t < k; ++t) {
            const double tp = static_cast<double>(hits[f * k + t]), p = static_cast<double>(predicted[f * k + t]);
            if (tp == 0) continue;
            const double precision = tp / p, recall = tp / static_cast<double>(out.positives), f1 = 2 * precision * recall / (precision + recall);
            best = std::max(best, f1);
            if (f1 > out.f1) {
                out.f1 = f1;
                out.feature = static_cast<int64_t>(f);
                out.threshold = thresholds[t];
                out.precision = precision;
                out.recall = recall;
            }
        }
        out.features_above_half += best > 0.5 ? 1 : 0;
    }
    return out;
}

}  // namespace pulsatrix
