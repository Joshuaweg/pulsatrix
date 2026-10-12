// Recipe: Is the shape real? (TDA-8, TDA-9, INT-5)
//
// Do a language model's representations of the days of the week, or the months, lie on a circle,
// as Engels et al. found in larger models? Persistent homology finds loops in any point cloud,
// including noise, so the recipe sets the tests before looking:
//   1. the universal null (Bobrowski and Skraba) flags the loop at the best layer;
//   2. the loop is longer-lived than in the same model with random weights (19 seeds);
//   3. and than in Gaussian clouds with the same mean and covariance (19 draws);
//   4. the words sit around the circle in calendar order, which label shuffles rarely give;
//   5. seven (or twelve) animal names in the same sentences show no loop.
// 2 to 4 compare the largest value over all layers with the same maximum under the null, so
// searching every layer doesn't inflate them.
//
// Usage: is_the_shape_real_recipe MODEL_DIR OUT_DIR [--device cpu|hip] [--set days|months]
//                                 [--null-seeds N] [--components K] [--smoke [--tokenizer PATH]]
//   MODEL_DIR holds model.safetensors (or shards), config.json and tokenizer.json, for example
//   ~/.cache/pulsatrix/golden/SmolLM2-135M. --smoke runs a few prompts on the tiny test model
//   (tests/fixtures/hf_tiny/llama) to check the plumbing; its numbers mean nothing.
//
// Docs: docs/recipes/mechanistic-interpretability/is_the_shape_real.md
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/null_model_baseline.hpp"
#include "pulsatrix/pca.hpp"
#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"
#include "pulsatrix/tda/significance.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "pulsatrix/viz/topology_views.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

const std::vector<std::string> kDays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
const std::vector<std::string> kMonths = {"January", "February", "March",     "April",   "May",      "June",
                                          "July",    "August",   "September", "October", "November", "December"};
const std::vector<std::string> kAnimals = {"cat", "dog", "horse", "rabbit", "sheep", "goat",
                                           "mouse", "cow", "pig", "duck", "fox", "bear"};

// Each ends where the word goes; the word is appended with a space.
const std::vector<std::string> kDayTemplates = {
    "Let's meet on", "The meeting was moved to", "I will see you on", "The store is closed on",
    "We always go swimming on", "My favorite day of the week is", "The package arrived on", "The concert is on",
    "See you next", "The deadline is this", "It happened last", "The class starts on",
    "He called me on", "The party is planned for", "They play football every", "The office opens again on",
    "I start my new job on", "The train leaves on", "Our flight is on", "The market is open every",
    "The exam is scheduled for", "Rain is expected on", "The game was postponed until", "I usually rest on",
    "The library is busy on", "Tickets go on sale this", "The museum is free on", "We clean the house every",
    "The doctor can see you on", "The results will be published on", "My sister visits us every", "The shop gets deliveries on",
    "The bus does not run on", "Choir practice is every", "The trash is collected on", "We watched the film on",
    "The new law takes effect on", "Her shift ends on", "The garden club meets on", "The report is due on"};
const std::vector<std::string> kMonthTemplates = {
    "I was born in", "The wedding is in", "School starts in", "It snowed heavily in",
    "We are moving in", "The festival takes place in", "Prices went up in", "The baby is due in",
    "The project ends in", "Our vacation is in", "The election was held in", "The bridge opened in",
    "My birthday is in", "The harvest begins in", "The company was founded in", "The trees bloom in",
    "The lease expires in", "We met in", "The conference is in", "The store reopens in",
    "Sales peaked in", "The river flooded in", "The album was released in", "The race is held every",
    "The semester ends in", "The tax deadline is in", "He retired in", "The museum closed in",
    "The road works start in", "The eclipse happened in", "Her contract ends in", "The team was formed in",
    "The new menu launches in", "The heat wave came in", "We planted the garden in", "The trial begins in",
    "The book comes out in", "Rents are reviewed every", "The park reopens in", "The storm hit in"};

struct Options {
    std::string model, out, device = "cpu", set = "days", tokenizer;
    int null_seeds = 19;
    int permutations = 9999;
    bool smoke = false;
};

// A portable generator, so every platform shuffles the same way.
struct Lcg {
    uint64_t state;
    uint64_t next(uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (state >> 33) % bound;
    }
};

// Last-token hidden states of every prompt at every residual-stream position, each template's mean
// over its words subtracted: positions × (prompts × hidden), prompts template-major.
using Clouds = std::vector<std::vector<float>>;

struct Corpus {
    std::vector<std::vector<int64_t>> ids;  // one per prompt, template-major
    int64_t words = 0, templates = 0;
};

Corpus Tokenize(const TextTokenizer& tok, const std::vector<std::string>& words, const std::vector<std::string>& templates,
                const Options& o, int64_t vocab) {
    Corpus c;
    c.words = static_cast<int64_t>(words.size());
    c.templates = static_cast<int64_t>(templates.size());
    for (const std::string& t : templates) {
        for (size_t wi = 0; wi < words.size(); ++wi) {
            const std::string& w = words[wi];
            Encoding e = tok.encode(t + " " + w);
            if (o.smoke) {
                // The test tokenizer splits the words, and "Monday" to "Sunday" all end in "day":
                // give each word a token of its own instead.
                e = tok.encode(t);
                e.ids.push_back(static_cast<int64_t>(1 + 5 * wi));
            } else {
                // The word must be one token, the prompt's last, so its hidden state is the word's.
                const Encoding alone = tok.encode(" " + w, /*add_special_tokens=*/false);
                if (alone.size() != 1 || e.ids.back() != alone.ids[0]) {
                    throw std::runtime_error("\"" + w + "\" isn't a single final token for this tokenizer");
                }
            }
            for (int64_t& id : e.ids) id %= vocab;  // only the tiny smoke-test model has a smaller vocabulary
            c.ids.push_back(e.ids);
        }
    }
    return c;
}

Clouds Capture(CausalLM& model, const Corpus& c, DeviceBackend* backend) {
    const int64_t positions = model.num_layers() + 1, h = model.config().hidden_size;
    const auto n = static_cast<size_t>(c.words * c.templates);
    Clouds clouds(static_cast<size_t>(positions), std::vector<float>(n * static_cast<size_t>(h)));
    std::vector<size_t> batch;  // prompt indices in the current forward
    int64_t L = 0;
    model.set_hidden_state_hook([&](int64_t position, const Tensor& hidden) {
        const std::vector<float> v = hidden.to_host_vector();
        for (size_t b = 0; b < batch.size(); ++b) {
            const auto last = v.begin() + static_cast<std::ptrdiff_t>((static_cast<int64_t>(b) * L + L - 1) * h);
            std::copy(last, last + h, clouds[static_cast<size_t>(position)].begin() + static_cast<std::ptrdiff_t>(batch[b] * h));
        }
        return hidden;
    });
    // One forward per template and prompt length: a template's prompts differ only in the word.
    for (int64_t t = 0; t < c.templates; ++t) {
        std::vector<size_t> todo;
        for (int64_t w = 0; w < c.words; ++w) todo.push_back(static_cast<size_t>(t * c.words + w));
        while (!todo.empty()) {
            L = static_cast<int64_t>(c.ids[todo.front()].size());
            batch.clear();
            std::vector<size_t> rest;
            std::vector<float> ids;
            for (size_t i : todo) {
                if (static_cast<int64_t>(c.ids[i].size()) != L) {
                    rest.push_back(i);
                    continue;
                }
                batch.push_back(i);
                ids.insert(ids.end(), c.ids[i].begin(), c.ids[i].end());
            }
            (void)model.forward(Tensor(Shape({static_cast<int64_t>(batch.size()), L}), backend, ids));
            todo = rest;
        }
    }
    model.set_hidden_state_hook(nullptr);
    // Subtract each template's mean over its words: what's left is how the words differ.
    for (std::vector<float>& cloud : clouds) {
        for (int64_t t = 0; t < c.templates; ++t) {
            for (int64_t d = 0; d < h; ++d) {
                double mean = 0;
                for (int64_t w = 0; w < c.words; ++w) mean += cloud[static_cast<size_t>((t * c.words + w) * h + d)];
                mean /= static_cast<double>(c.words);
                for (int64_t w = 0; w < c.words; ++w) cloud[static_cast<size_t>((t * c.words + w) * h + d)] -= static_cast<float>(mean);
            }
        }
    }
    return clouds;
}

// The principal components the topology sees (--components). 5 is the setting fixed before the
// first run; the circle itself lies in the first two.
int64_t g_components = 5;

// One layer's cloud, reduced to its top principal components, and its H1.
struct Layer {
    Pca pca;
    std::vector<float> scores;       // (points, g_components)
    int64_t points = 0;
    RipsResult rips;
    double ratio = 0;                // the largest death / birth in H1: scale-free
    std::vector<double> angles;      // each word's centroid in the PC1-PC2 plane
};

double MaxRatio(const PersistenceDiagram& d) {
    double best = 0;
    for (const PersistencePair& p : d.pairs) {
        if (!p.essential() && p.birth > 0) best = std::max(best, p.death / p.birth);
    }
    return best;
}

double RatioOf(const Tensor& points) {
    DistanceOptions dopt;
    dopt.center = true;
    return MaxRatio(ComputeRipsPersistence(PairwiseDistances(points, dopt)).diagrams[1]);
}

Layer Analyze(const std::vector<float>& cloud, int64_t words, int64_t h, CPUBackend* cpu) {
    const auto n = static_cast<int64_t>(cloud.size()) / h;
    Layer l;
    const Tensor x(Shape({n, h}), cpu, cloud);
    l.pca = FitPca(x, g_components);
    const Tensor scores = PcaProject(l.pca, x);
    l.scores = scores.to_host_vector();
    l.points = n;
    DistanceOptions dopt;
    dopt.center = true;
    l.rips = ComputeRipsPersistence(PairwiseDistances(scores, dopt));
    l.ratio = MaxRatio(l.rips.diagrams[1]);
    const std::vector<float>& s = l.scores;
    for (int64_t w = 0; w < words; ++w) {
        double cx = 0, cy = 0;
        for (int64_t i = w; i < n; i += words) {
            cx += s[static_cast<size_t>(i * g_components)];
            cy += s[static_cast<size_t>(i * g_components + 1)];
        }
        l.angles.push_back(std::atan2(cy, cx));
    }
    return l;
}

// How many calendar neighbors (word i and i+1, and the last and the first) are neighbors around the
// circle, after relabeling word w as label[w]. 0 to the number of words.
int OrderScore(const std::vector<double>& angles, const std::vector<int>& label) {
    const int W = static_cast<int>(angles.size());
    std::vector<int> order(static_cast<size_t>(W));
    for (int i = 0; i < W; ++i) order[static_cast<size_t>(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return angles[static_cast<size_t>(a)] < angles[static_cast<size_t>(b)]; });
    std::vector<int> pos(static_cast<size_t>(W));
    for (int k = 0; k < W; ++k) pos[static_cast<size_t>(label[static_cast<size_t>(order[static_cast<size_t>(k)])])] = k;
    int score = 0;
    for (int i = 0; i < W; ++i) {
        const int gap = (pos[static_cast<size_t>(i)] - pos[static_cast<size_t>((i + 1) % W)] + W) % W;
        score += (gap == 1 || gap == W - 1) ? 1 : 0;
    }
    return score;
}

// The universal null at one layer: the smallest p-value and whether any loop passes alpha.
struct Universal {
    double p = std::numeric_limits<double>::quiet_NaN();
    bool significant = false;
    UniversalNullResult result;
};

Universal UniversalAt(const PersistenceDiagram& h1, double alpha) {
    Universal u;
    try {
        u.result = UniversalNullTest(h1, alpha);
    } catch (const std::invalid_argument&) {
        return u;  // too few loops to test (only in --smoke)
    }
    for (const FeatureSignificance& f : u.result.features) {
        if (std::isnan(u.p) || f.p_value < u.p) u.p = f.p_value;
        u.significant = u.significant || f.significant;
    }
    return u;
}

std::string Fixed(double v, int digits = 3) {
    if (std::isnan(v)) return "n/a";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}

std::string Sci(double v) {
    if (std::isnan(v)) return "n/a";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2g", v);
    return buf;
}

void Write(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
    if (!out) throw std::runtime_error("can't write " + path);
}

int Run(DeviceBackend* backend, const Options& o) {
    CPUBackend cpu;
    std::filesystem::create_directories(o.out);
    const TextTokenizer tok = LoadTokenizerJson(o.tokenizer.empty() ? o.model + "/tokenizer.json" : o.tokenizer);
    std::unique_ptr<CausalLM> model = LoadCausalLM(o.model, backend);
    const int64_t positions = model->num_layers() + 1, h = model->config().hidden_size;
    const bool days = o.set == "days";
    std::vector<std::string> words = days ? kDays : kMonths;
    std::vector<std::string> templates = days ? kDayTemplates : kMonthTemplates;
    std::vector<std::string> control(kAnimals.begin(), kAnimals.begin() + static_cast<std::ptrdiff_t>(words.size()));
    if (o.smoke) templates.resize(6);
    const auto W = static_cast<int64_t>(words.size());
    const double alpha = 0.05 / static_cast<double>(positions);  // Bonferroni over the layers

    const Corpus corpus = Tokenize(tok, words, templates, o, model->config().vocab_size);
    const Corpus control_corpus = Tokenize(tok, control, templates, o, model->config().vocab_size);
    std::cout << "Model: " << o.model << " (" << model->num_layers() << " blocks, hidden " << h << ")\n"
              << "Points per layer: " << W << " words x " << templates.size() << " sentences = " << W * static_cast<int64_t>(templates.size())
              << ", each sentence's mean subtracted, then the top " << g_components << " principal components\n";
    if (o.smoke) std::cout << "--smoke: the tiny test model and a handful of prompts; the numbers check the plumbing only\n";

    // The trained model.
    const Clouds trained = Capture(*model, corpus, backend);
    std::vector<Layer> layers;
    std::vector<Universal> universal;
    for (int64_t p = 0; p < positions; ++p) {
        layers.push_back(Analyze(trained[static_cast<size_t>(p)], W, h, &cpu));
        universal.push_back(UniversalAt(layers.back().rips.diagrams[1], alpha));
    }
    std::vector<int> identity(static_cast<size_t>(W));
    for (int i = 0; i < W; ++i) identity[static_cast<size_t>(i)] = i;
    std::vector<int> order(static_cast<size_t>(positions));
    for (int64_t p = 0; p < positions; ++p) order[static_cast<size_t>(p)] = OrderScore(layers[static_cast<size_t>(p)].angles, identity);
    int64_t best = 0;
    for (int64_t p = 1; p < positions; ++p) {
        if (layers[static_cast<size_t>(p)].ratio > layers[static_cast<size_t>(best)].ratio) best = p;
    }
    auto max_ratio = [&](const std::vector<double>& r) { return *std::max_element(r.begin(), r.end()); };
    std::vector<double> trained_ratio;
    for (const Layer& l : layers) trained_ratio.push_back(l.ratio);

    // Random networks: the same model, each weight tensor redrawn at its own scale.
    std::vector<std::vector<double>> random_ratio;  // seed × layer
    Layer random_best;
    {
        ParameterSnapshot saved(*model);
        for (int s = 1; s <= o.null_seeds; ++s) {
            ReinitializeParameters(*model, static_cast<uint64_t>(s));
            const Clouds r = Capture(*model, corpus, backend);
            std::vector<double> ratios;
            for (int64_t p = 0; p < positions; ++p) {
                if (s == 1 && p == best) {
                    random_best = Analyze(r[static_cast<size_t>(p)], W, h, &cpu);
                    ratios.push_back(random_best.ratio);
                } else {
                    ratios.push_back(Analyze(r[static_cast<size_t>(p)], W, h, &cpu).ratio);
                }
            }
            random_ratio.push_back(ratios);
            saved.restore();
        }
    }

    // Gaussian clouds with each layer's mean and covariance, in the same principal components.
    std::vector<std::vector<double>> gaussian_ratio(static_cast<size_t>(o.null_seeds));
    for (int64_t p = 0; p < positions; ++p) {
        const Layer& l = layers[static_cast<size_t>(p)];
        for (int g = 0; g < o.null_seeds; ++g) {
            const Tensor scores(Shape({l.points, g_components}), &cpu, l.scores);
            gaussian_ratio[static_cast<size_t>(g)].push_back(RatioOf(GaussianBaseline(scores, l.points, static_cast<uint32_t>(g + 1))));
        }
    }

    // Calendar order: shuffle which word is which, and take the best layer's score each time.
    const int observed_order = *std::max_element(order.begin(), order.end());
    Lcg rng{2024};
    int at_least = 0;
    for (int k = 0; k < o.permutations; ++k) {
        std::vector<int> label = identity;
        for (int64_t i = W - 1; i > 0; --i) std::swap(label[static_cast<size_t>(i)], label[rng.next(static_cast<uint64_t>(i + 1))]);
        int best_score = 0;
        for (const Layer& l : layers) best_score = std::max(best_score, OrderScore(l.angles, label));
        at_least += best_score >= observed_order ? 1 : 0;
    }
    const double p_order = (1.0 + at_least) / (1.0 + o.permutations);

    // The control words, through the same steps.
    const Clouds animals = Capture(*model, control_corpus, backend);
    std::vector<double> control_ratio;
    std::vector<Universal> control_universal;
    for (int64_t p = 0; p < positions; ++p) {
        const Layer l = Analyze(animals[static_cast<size_t>(p)], W, h, &cpu);
        control_ratio.push_back(l.ratio);
        control_universal.push_back(UniversalAt(l.rips.diagrams[1], alpha));
    }

    // Per layer (uncorrected, for the heatmap and the table).
    std::cout << "\nlayer  H1 max death/birth  universal p  vs random nets  vs Gaussian  calendar neighbors  animals' ratio\n";
    HeatmapDocument heat;
    heat.title = "-log10 p by layer (uncorrected)";
    heat.rows = 4;
    heat.cols = positions;
    heat.row_labels = {"universal null", "random nets", "Gaussian", "animals"};
    std::vector<float> rows[4];
    for (int64_t p = 0; p < positions; ++p) {
        std::vector<double> rr, gg;
        for (const auto& s : random_ratio) rr.push_back(s[static_cast<size_t>(p)]);
        for (const auto& s : gaussian_ratio) gg.push_back(s[static_cast<size_t>(p)]);
        const double p_random = BaselinePValue(trained_ratio[static_cast<size_t>(p)], rr);
        const double p_gauss = BaselinePValue(trained_ratio[static_cast<size_t>(p)], gg);
        const auto lg = [](double v) { return std::isnan(v) ? 0.0f : static_cast<float>(-std::log10(std::max(v, 1e-300))); };
        rows[0].push_back(lg(universal[static_cast<size_t>(p)].p));
        rows[1].push_back(lg(p_random));
        rows[2].push_back(lg(p_gauss));
        rows[3].push_back(lg(control_universal[static_cast<size_t>(p)].p));
        heat.col_labels.push_back(std::to_string(p));
        std::printf("%5lld  %18s  %11s  %14s  %11s  %12d of %-3lld  %14s%s\n", static_cast<long long>(p),
                    Fixed(trained_ratio[static_cast<size_t>(p)]).c_str(), Sci(universal[static_cast<size_t>(p)].p).c_str(),
                    Fixed(p_random, 2).c_str(), Fixed(p_gauss, 2).c_str(), order[static_cast<size_t>(p)], static_cast<long long>(W),
                    Fixed(control_ratio[static_cast<size_t>(p)]).c_str(), p == best ? "  <- largest loop" : "");
    }
    for (const auto& r : rows) heat.values.insert(heat.values.end(), r.begin(), r.end());

    // The verdict, family-wise: each test compares the maximum over layers with its null maximum.
    std::vector<double> random_max, gauss_max;
    for (const auto& s : random_ratio) random_max.push_back(max_ratio(s));
    for (const auto& s : gaussian_ratio) gauss_max.push_back(max_ratio(s));
    const double p_random = BaselinePValue(max_ratio(trained_ratio), random_max);
    const double p_gauss = BaselinePValue(max_ratio(trained_ratio), gauss_max);
    bool control_loop = false;
    for (const Universal& u : control_universal) control_loop = control_loop || u.significant;
    const Universal& u = universal[static_cast<size_t>(best)];
    struct Test {
        std::string what;
        bool pass;
        std::string detail;
    };
    const std::vector<Test> tests = {
        {"universal null flags the loop at layer " + std::to_string(best), u.significant,
         "smallest p " + Sci(u.p) + ", threshold " + Sci(u.result.threshold)},
        {"longer-lived than in random networks", p_random <= 0.05,
         "p = " + Fixed(p_random, 3) + " (" + std::to_string(o.null_seeds) + " seeds, max over layers)"},
        {"longer-lived than in Gaussian clouds", p_gauss <= 0.05,
         "p = " + Fixed(p_gauss, 3) + " (" + std::to_string(o.null_seeds) + " draws per layer, max over layers)"},
        {"words in calendar order around the circle", p_order <= 0.05,
         std::to_string(observed_order) + " of " + std::to_string(W) + " neighbors, p = " + Sci(p_order) + " (" +
             std::to_string(o.permutations) + " label shuffles)"},
        {"no loop for the animal names", !control_loop, control_loop ? "a layer has a significant loop" : "no layer has a significant loop"},
    };
    bool all = true;
    std::cout << "\nTests (fixed before the run):\n";
    for (const Test& t : tests) {
        std::cout << "  [" << (t.pass ? "pass" : "FAIL") << "] " << t.what << ": " << t.detail << "\n";
        all = all && t.pass;
    }
    std::cout << "Verdict: " << (all ? "a circle at layer " + std::to_string(best) : std::string("no circle; see the failed tests"))
              << "\n";

    // Figures.
    const std::string where = (days ? "weekday" : "month") + std::string(" tokens, layer ") + std::to_string(best);
    PersistenceDiagramDocument ph = ToPersistenceDiagramDocument(layers[static_cast<size_t>(best)].rips, "Trained model, " + where);
    ph.source = o.model;
    ph.num_points = layers[static_cast<size_t>(best)].points;
    if (!u.result.features.empty()) AttachSignificance(ph, 1, u.result);
    PersistenceDiagramDocument ph_random = ToPersistenceDiagramDocument(random_best.rips, "Random weights, " + where);
    ph_random.num_points = ph.num_points;
    const Universal ur = UniversalAt(random_best.rips.diagrams[1], alpha);
    if (!ur.result.features.empty()) AttachSignificance(ph_random, 1, ur.result);
    PersistenceViewOptions h1;
    h1.dimensions = {1};
    Write(o.out + "/persistence.json", ToJson(ph));
    Write(o.out + "/persistence_diagram.svg", RenderPersistenceDiagramSvg(ph, h1));
    Write(o.out + "/barcode.svg", RenderBarcodeSvg(ph, h1));
    Write(o.out + "/persistence_diagram_random.svg", RenderPersistenceDiagramSvg(ph_random, h1));

    std::vector<std::string> groups, labels;
    for (const std::string& t : templates) {
        for (const std::string& w : words) {
            groups.push_back(w);
            labels.push_back(t + " " + w);
        }
    }
    auto projection = [&](const Layer& l, const std::string& title, const std::string& file) {
        const ProjectionDocument doc =
            MakeProjectionDocument(l.pca, Tensor(Shape({l.points, g_components}), &cpu, l.scores), labels, groups, {}, title);
        Write(o.out + "/" + file + ".json", ToJson(doc));
        Write(o.out + "/" + file + ".svg", RenderProjectionSvg(doc));
    };
    projection(layers[static_cast<size_t>(best)], "Trained model, " + where, "projection");
    projection(layers[0], "Trained model, the embeddings (layer 0)", "projection_layer0");
    projection(random_best, "Random weights, " + where, "projection_random");
    Write(o.out + "/tests_by_layer.json", ToJson(heat));
    Write(o.out + "/tests_by_layer.svg", RenderHeatmapSvg(heat));
    std::cout << "Wrote the figures and documents to " << o.out << "/\n";
    return 0;
}

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "is_the_shape_real_recipe: " << problem << "\n"
              << "usage: is_the_shape_real_recipe MODEL_DIR OUT_DIR [--device cpu|hip] [--set days|months]\n"
              << "                                [--null-seeds N] [--components K] [--smoke [--tokenizer PATH]]\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--device") o.device = value();
        else if (a == "--set") o.set = value();
        else if (a == "--null-seeds") o.null_seeds = std::stoi(value());
        else if (a == "--components") g_components = std::stoll(value());
        else if (a == "--tokenizer") o.tokenizer = value();
        else if (a == "--smoke") o.smoke = true;
        else if (!a.empty() && a[0] == '-') Usage("unknown option " + a);
        else positional.push_back(a);
    }
    if (positional.size() != 2) Usage("needs MODEL_DIR and OUT_DIR");
    if (o.set != "days" && o.set != "months") Usage("--set needs days or months");
    if (o.null_seeds < 1) Usage("--null-seeds must be at least 1");
    if (g_components < 2) Usage("--components must be at least 2");
    o.model = positional[0];
    o.out = positional[1];
    if (o.smoke) {
        o.null_seeds = std::min(o.null_seeds, 2);
        o.permutations = 99;
    }
    try {
        if (o.device == "cpu") {
            CPUBackend cpu;
            return Run(&cpu, o);
        }
#ifdef PULSATRIX_DEMO_WITH_HIP
        if (o.device == "hip") {
            HIPBackend hip;
            return Run(&hip, o);
        }
#endif
        Usage("device \"" + o.device + "\" is not in this build");
    } catch (const std::exception& e) {
        std::cerr << "is_the_shape_real_recipe: " << e.what() << "\n";
        return 1;
    }
}
