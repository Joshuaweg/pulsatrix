/** @file hpo_genotype.hpp
 *  @brief Decodes a real-valued genotype (a unit-hypercube point, one gene per parameter,
 *         each in [0, 1]) into a full hyperparameter Configuration -- the encoding this
 *         campaign's GA operators (crossover.hpp/mutation.hpp, all defined over
 *         std::vector<double>) evolve directly, decoded only when a genotype needs to be
 *         evaluated.
 *  @ingroup evolutionary
 *  @note Reuses the same unit-hypercube convention
 *        campaign_exai_dl_library_hyperparameter_optimization's own GP-BO surrogate
 *        established (gp_bo.hpp's UnitCubeToConfiguration) for Continuous/LogUniform
 *        parameters, but -- unlike that function, which deliberately throws on Integer/
 *        Categorical since vanilla GP-BO's kernel cannot handle them -- this decoder handles
 *        every ParameterKind, since this campaign's own GA operators have no such
 *        restriction (crossover/mutation over a real-valued vector composes with Integer and
 *        Categorical genes exactly as easily as Continuous ones; only the final decode step
 *        differs per kind).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/search_space.hpp"

namespace pulsatrix {

/**
 * @brief Decodes a unit-hypercube genotype into a Configuration using space's own declared
 *        bounds: Continuous linearly, LogUniform geometrically (both identical to
 *        gp_bo.hpp's UnitCubeToConfiguration); Integer by linear interpolation then rounding
 *        to the nearest integer (std::round -- ties away from zero, portable); Categorical by
 *        linear interpolation into an index, floored, clamped to the last category (handles
 *        the gene == 1.0 boundary, which would otherwise floor to one-past-the-end).
 * @throws std::invalid_argument if genotype.size() != space.size(), or any gene is outside
 *         [0, 1].
 */
inline Configuration DecodeGenotype(const SearchSpace& space, const std::vector<double>& genotype) {
    if (genotype.size() != space.size()) {
        throw std::invalid_argument("DecodeGenotype: genotype.size() must equal space.size()");
    }
    Configuration config;
    for (size_t i = 0; i < space.parameters().size(); ++i) {
        const auto& spec = space.parameters()[i];
        double gene = genotype[i];
        if (gene < 0.0 || gene > 1.0) {
            throw std::invalid_argument("DecodeGenotype: every gene must be in [0, 1]");
        }
        switch (spec.kind) {
            case ParameterKind::Continuous:
                config[spec.name] = spec.lower + gene * (spec.upper - spec.lower);
                break;
            case ParameterKind::LogUniform: {
                double log_lower = std::log(spec.lower);
                double log_upper = std::log(spec.upper);
                config[spec.name] = std::exp(log_lower + gene * (log_upper - log_lower));
                break;
            }
            case ParameterKind::Integer: {
                double interpolated = spec.lower + gene * (spec.upper - spec.lower);
                int64_t rounded = static_cast<int64_t>(std::round(interpolated));
                rounded = std::min(rounded, static_cast<int64_t>(spec.upper));
                rounded = std::max(rounded, static_cast<int64_t>(spec.lower));
                config[spec.name] = rounded;
                break;
            }
            case ParameterKind::Categorical: {
                size_t num_categories = spec.categories.size();
                size_t index = static_cast<size_t>(gene * static_cast<double>(num_categories));
                index = std::min(index, num_categories - 1);
                config[spec.name] = spec.categories[index];
                break;
            }
        }
    }
    return config;
}

}  // namespace pulsatrix
