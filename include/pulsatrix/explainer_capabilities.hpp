/** @file explainer_capabilities.hpp
 *  @brief What each explainer answers: the nine question categories (Liao, Gruen and Miller, CHI
 *         2020), the four reasons to explain (Adadi and Berrada, 2018), and a machine-readable
 *         declaration of every explainer's question, scope, model access and inputs (XAI-1).
 *  @ingroup interpretability
 *
 * Pulsatrix's explainers are separate classes and free functions with different signatures, so the
 * declarations live in one catalog, ExplainerCatalog(). The classes with an `explain()` method also
 * return their own entry from a static `capabilities()`.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/json.hpp"

namespace pulsatrix {

/** @brief The nine question categories of Liao, Gruen and Miller's XAI question bank (CHI 2020). */
enum class XaiQuestion {
    /// What data the model was trained on or sees: dataset statistics, the training examples
    /// nearest an input.
    Input,
    /// What the model's output is and means: its schema, labels and scope.
    Output,
    /// How well the model does: accuracy, calibration, error by slice, uncertainty.
    Performance,
    /// How the model works overall: its general logic, the features and concepts it uses.
    HowGlobal,
    /// Why this input gets this prediction.
    Why,
    /// Why this input gets this prediction instead of another one.
    WhyNot,
    /// What happens to the prediction if the input changes in a given way.
    WhatIf,
    /// What change to the input would give a different, chosen prediction.
    HowToBeThat,
    /// What can change, or must stay, for the prediction to remain the same.
    HowToStillBeThis
};

/** @brief Adadi and Berrada's four reasons to explain a model (IEEE Access, 2018). */
enum class XaiReason {
    /// Justify a decision to someone it affects or who must sign it off.
    Justify,
    /// Control the model: find its flaws and failure modes before they do harm.
    Control,
    /// Improve the model: find what to change in its data, features or training.
    Improve,
    /// Discover something new about the domain from what the model learned.
    Discover
};

/** @brief Whether an explainer explains one prediction or the model as a whole. */
enum class ExplanationScope {
    /// One input at a time (a batch is explained row by row).
    Local,
    /// The model, a layer or a dataset.
    Global
};

/** @brief The least an explainer needs from the model, from least to most access. */
enum class ModelAccess {
    /// Only a prediction function, `Tensor(const Tensor&)`: works on any model.
    BlackBox,
    /// The weights alone, without running the model.
    Weights,
    /// Activations at a layer, collected beforehand or through a hook.
    Activations,
    /// Gradients through a pulsatrix network (an ExplainerContext or a Module).
    Gradients,
    /// An LRP rule at every layer of a pulsatrix network.
    Relevance
};

/** @brief What the explainer's input has to be. */
enum class InputKind {
    /// Any tensor.
    Any,
    /// A flat feature vector, one value per named feature.
    Tabular,
    /// Images: `(N, C, H, W)`.
    Image,
    /// Token ids into a language model.
    Tokens,
    /// Amino-acid sequences into a protein language model.
    Protein,
    /// Facts and rules in the Datalog engine.
    Symbolic,
    /// Representations: an `(N, d)` matrix of activations or embeddings.
    Representation
};

/** @brief One explainer's declaration: what it answers and what it needs. */
struct ExplainerCapabilities {
    /** @brief A stable identifier, the class or function name: "IntegratedGradients". */
    std::string name;
    /** @brief The header to include: "pulsatrix/integrated_gradients.hpp". */
    std::string header;
    /** @brief The calls that run it: "IntegratedGradients::explain". */
    std::vector<std::string> entry_points;
    /** @brief One sentence on what it computes. */
    std::string summary;
    /** @brief The questions it answers, the main one first. */
    std::vector<XaiQuestion> questions;
    ExplanationScope scope = ExplanationScope::Local;
    ModelAccess access = ModelAccess::BlackBox;
    InputKind input = InputKind::Any;
    /** @brief It returns an Attribution with one value per input element, so the metrics in
     *         explanation_metrics.hpp (deletion, insertion, randomization, sparseness) apply. */
    bool attribution = false;
    /** @brief The attributions add up to the output, minus a baseline's where there is one:
     *         LRP's conservation, Integrated Gradients' completeness, Shapley values' efficiency.
     *         It can be checked on every explanation (ComputeConservation()), and LRP's ε rules keep
     *         it only approximately. */
    bool complete = false;
    /** @brief It compares a model with the one it was fine-tuned from, so it needs both. */
    bool needs_base_model = false;
    /** @brief It scores the model against true labels (the Performance metrics, conformal
     *         calibration), so it needs a labeled dataset (ModelTraits::has_labels). */
    bool needs_labels = false;
    /** @brief The most input features it can take, or 0 for no limit: KernelSHAP enumerates every
     *         coalition, so it takes at most 20. */
    int64_t max_features = 0;

    /** @brief Whether @p question is among questions. */
    [[nodiscard]] bool answers(XaiQuestion question) const;
};

/** @name Names. Lower-case snake case, as in reports and JSON: "how_global", "justify". */
///@{
[[nodiscard]] std::string_view to_string(XaiQuestion question);
[[nodiscard]] std::string_view to_string(XaiReason reason);
[[nodiscard]] std::string_view to_string(ExplanationScope scope);
[[nodiscard]] std::string_view to_string(ModelAccess access);
[[nodiscard]] std::string_view to_string(InputKind input);
///@}

/** @name Parsing. The inverse of to_string(); std::nullopt for any other text. */
///@{
[[nodiscard]] std::optional<XaiQuestion> ParseXaiQuestion(std::string_view text);
[[nodiscard]] std::optional<XaiReason> ParseXaiReason(std::string_view text);
///@}

/** @brief The question as Liao et al. phrase it, for example "Why is this instance given this
 *         prediction?". */
[[nodiscard]] std::string_view QuestionText(XaiQuestion question);
/** @brief One sentence on the reason. */
[[nodiscard]] std::string_view ReasonText(XaiReason reason);

/** @brief All nine questions, in the enum's order. */
[[nodiscard]] const std::vector<XaiQuestion>& AllXaiQuestions();
/** @brief All four reasons, in the enum's order. */
[[nodiscard]] const std::vector<XaiReason>& AllXaiReasons();

/** @brief Every explainer pulsatrix has, in a fixed order. */
[[nodiscard]] const std::vector<ExplainerCapabilities>& ExplainerCatalog();

/** @brief The catalog entry named @p name.
 *  @throws std::invalid_argument if there is none. */
[[nodiscard]] const ExplainerCapabilities& ExplainerCapabilitiesOf(std::string_view name);

/** @brief The explainers that answer @p question: those that list it first, then the rest, each
 *         group in catalog order. Empty for a question pulsatrix has no explainer for yet. */
[[nodiscard]] std::vector<const ExplainerCapabilities*> ExplainersFor(XaiQuestion question);

/** @brief The declaration as a JSON object, with the enums as their to_string() names. */
[[nodiscard]] JsonValue ToJson(const ExplainerCapabilities& capabilities);
/** @brief The whole catalog: a `pulsatrix.explainer_catalog.v1` document. */
[[nodiscard]] JsonValue ExplainerCatalogDocument();

}  // namespace pulsatrix
