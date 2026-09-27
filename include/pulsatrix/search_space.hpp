/** @file search_space.hpp
 *  @brief Typed hyperparameter search-space description: named parameters, each continuous,
 *         log-uniform, integer, or categorical.
 *  @ingroup hyperparameter_optimization
 *  @note Recon (2026-09-27, campaign_exai_dl_library_hyperparameter_optimization activation):
 *        no SearchSpace/Trial/HPO type of any kind existed anywhere in this codebase before
 *        this mission -- genuinely new, purely additive scaffolding, zero changes to
 *        Module/Tensor/DeviceBackend/MetricsSink.
 */
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace pulsatrix {

/** @brief Which family of values a named parameter draws from. */
enum class ParameterKind { Continuous, LogUniform, Integer, Categorical };

/**
 * @brief One named parameter's description: its kind plus the bounds/categories that kind
 *        needs. Only the fields relevant to `kind` are meaningful (e.g. `categories` is
 *        empty/unused for `Continuous`) -- this is a description, not a union, since a
 *        `SearchSpace`'s own accessors (below) are the only place callers read it back.
 */
struct ParameterSpec {
    std::string name;
    ParameterKind kind;
    double lower = 0.0;
    double upper = 0.0;
    std::vector<std::string> categories;
};

/**
 * @brief A concrete value for one parameter -- a `double` (Continuous/LogUniform), an
 *        `int64_t` (Integer), or a `std::string` (Categorical, naming one of that
 *        parameter's categories).
 */
using ConfigValue = std::variant<double, int64_t, std::string>;

/** @brief A concrete hyperparameter configuration: parameter name -> concrete value. */
using Configuration = std::map<std::string, ConfigValue>;

/**
 * @brief Describes a hyperparameter search space as an ordered list of named, typed
 *        parameters. Every HPO algorithm (grid/random search, GP-BO, TPE, Hyperband/ASHA)
 *        consumes a `SearchSpace` to know what it may propose; this type itself has no
 *        sampling logic (that is each algorithm's own job, e.g. `RandomSample`/`GridSample`).
 */
class SearchSpace {
public:
    /**
     * @brief Adds a continuous parameter sampled from [lower, upper].
     * @throws std::invalid_argument if name is already used in this search space, or
     *         !(lower < upper).
     */
    void AddContinuous(std::string name, double lower, double upper) {
        ValidateNewName(name);
        if (!(lower < upper)) {
            throw std::invalid_argument("SearchSpace::AddContinuous: lower must be < upper");
        }
        parameters_.push_back(ParameterSpec{std::move(name), ParameterKind::Continuous, lower,
                                             upper, {}});
    }

    /**
     * @brief Adds a log-uniform parameter sampled from [lower, upper] on a log scale (e.g.
     *        learning rates, where 0.001 and 0.01 should be equally likely orders of
     *        magnitude, not equally likely absolute distances).
     * @throws std::invalid_argument if name is already used, lower <= 0, or !(lower < upper).
     */
    void AddLogUniform(std::string name, double lower, double upper) {
        ValidateNewName(name);
        if (lower <= 0.0) {
            throw std::invalid_argument("SearchSpace::AddLogUniform: lower must be > 0");
        }
        if (!(lower < upper)) {
            throw std::invalid_argument("SearchSpace::AddLogUniform: lower must be < upper");
        }
        parameters_.push_back(ParameterSpec{std::move(name), ParameterKind::LogUniform, lower,
                                             upper, {}});
    }

    /**
     * @brief Adds an integer parameter sampled from [lower, upper] (inclusive both ends).
     * @throws std::invalid_argument if name is already used, or lower > upper.
     */
    void AddInteger(std::string name, int64_t lower, int64_t upper) {
        ValidateNewName(name);
        if (lower > upper) {
            throw std::invalid_argument("SearchSpace::AddInteger: lower must be <= upper");
        }
        parameters_.push_back(ParameterSpec{std::move(name), ParameterKind::Integer,
                                             static_cast<double>(lower),
                                             static_cast<double>(upper), {}});
    }

    /**
     * @brief Adds a categorical parameter, one of categories.
     * @throws std::invalid_argument if name is already used, or categories is empty.
     */
    void AddCategorical(std::string name, std::vector<std::string> categories) {
        ValidateNewName(name);
        if (categories.empty()) {
            throw std::invalid_argument("SearchSpace::AddCategorical: categories must be non-empty");
        }
        parameters_.push_back(
            ParameterSpec{std::move(name), ParameterKind::Categorical, 0.0, 0.0, std::move(categories)});
    }

    /** @brief Every parameter, in the order added. */
    [[nodiscard]] const std::vector<ParameterSpec>& parameters() const { return parameters_; }

    /** @brief Number of parameters in this search space. */
    [[nodiscard]] size_t size() const { return parameters_.size(); }

    /**
     * @brief Looks up a parameter by name.
     * @throws std::out_of_range if no parameter with that name exists.
     */
    [[nodiscard]] const ParameterSpec& Get(const std::string& name) const {
        for (const auto& p : parameters_) {
            if (p.name == name) {
                return p;
            }
        }
        throw std::out_of_range("SearchSpace::Get: no parameter named '" + name + "'");
    }

    /** @brief Whether a parameter with this name exists. */
    [[nodiscard]] bool Contains(const std::string& name) const {
        for (const auto& p : parameters_) {
            if (p.name == name) {
                return true;
            }
        }
        return false;
    }

private:
    void ValidateNewName(const std::string& name) const {
        if (Contains(name)) {
            throw std::invalid_argument("SearchSpace: parameter name '" + name + "' already exists");
        }
    }

    std::vector<ParameterSpec> parameters_;
};

}  // namespace pulsatrix
