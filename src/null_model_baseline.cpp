#include "pulsatrix/null_model_baseline.hpp"

#include <cmath>

#include "portable_random.hpp"

namespace pulsatrix {

void ReinitializeParameters(Module& model, uint64_t seed, const std::string& prefix) {
    PortableRng rng{seed};
    for (const NamedParamRef& p : model.named_parameters()) {
        if (!prefix.empty() && p.name != prefix && p.name.rfind(prefix + ".", 0) != 0) {
            continue;
        }
        Tensor& t = *p.ref.value;
        std::vector<float> v = t.to_host_vector();
        double mean = 0.0, sq = 0.0;
        for (float e : v) mean += e;
        mean /= static_cast<double>(v.size());
        for (float e : v) sq += (e - mean) * (e - mean);
        const double std_dev = std::sqrt(sq / static_cast<double>(v.size()));
        for (float& e : v) e = static_cast<float>(std_dev * rng.gaussian());
        t = Tensor(t.shape(), t.backend(), v, t.device());
    }
}

AttributionNullReport NullModelBaseline(Module& model, const ExplainFn& explain, const Tensor& input, uint64_t seed) {
    NullModelComparison<Attribution> c = NullModelBaseline(model, [&] { return explain(input); }, seed);
    std::vector<float> a = c.trained.values.to_host_vector(), b = c.null_model.values.to_host_vector();
    for (float& v : a) v = std::fabs(v);
    for (float& v : b) v = std::fabs(v);
    const float similarity = SpearmanRankCorrelation(a, b);
    return {std::move(c.trained), std::move(c.null_model), similarity};
}

}  // namespace pulsatrix
