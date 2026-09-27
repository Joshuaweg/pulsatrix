/** @file neuro_symbolic_datalog_bridge.hpp
 *  @brief Wires a real neural predicate (`LinearModule` + sigmoid, reusing Phase 1 Mission 2's
 *         `ToyKnowledgeBase` pattern) into the weighted Datalog engine as one base fact's
 *         weight, on a small diamond-graph transitive-closure toy program (Mission 0/1's own
 *         `ancestor` shape). Computes the gradient of a derived query fact's weight w.r.t. the
 *         neural predicate's output (via forward-mode AD over the provenance semiring, see
 *         datalog_dual_semiring.hpp) and, transitively, the predicate's `LinearModule`
 *         parameters (via that module's own real `backward()`, mirroring `ToyKnowledgeBase`'s
 *         hand-chained-`Module::backward()` precedent). Phase 3 Mission 2 of
 *         campaign_exai_dl_library_neuro_symbolic (Neural-Predicate Integration).
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/datalog_dual_semiring.hpp"
#include "pulsatrix/datalog_lrp.hpp"
#include "pulsatrix/datalog_rule.hpp"
#include "pulsatrix/datalog_semiring.hpp"
#include "pulsatrix/datalog_weighted_fact_database.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix::datalog {

/**
 * @brief The two numbers a forward pass through the bridge produces: the derived query fact's
 *        real-valued weight, and its exact partial derivative w.r.t. the neural predicate's
 *        (sigmoid-squashed) output -- both computed in the same `DualSemiring<double>`
 *        evaluation pass (see datalog_dual_semiring.hpp).
 */
struct NeuralPredicateQueryResult {
    double query_weight;             ///< ancestor(a,d)'s derived weight.
    double grad_wrt_predicate_output;  ///< d(query_weight) / d(edge(a,b)'s weight, i.e. the sigmoid output).
};

/**
 * @brief The result of one `NeuralPredicateDatalogBridge::propagate_relevance()` call --
 *        Phase 3 Mission 3's own deliverable (LRP for the Datalog/provenance-semiring
 *        circuit), extended end-to-end through the neural predicate's own `Module` chain.
 * @note **Relevance-termination decision (Stage 3 design question 2, Mission 3)**: relevance
 *       does NOT stop at `edge(a,b)` (the neural-predicate-weighted base fact) -- it
 *       continues into the predicate's own `LinearModule`+sigmoid chain via composition,
 *       reusing Phase 1's existing `propagate_relevance` methods (no new rule): `datalog_lrp`'s
 *       own `propagate_relevance_weighted` treats `edge(a,b)` as an ordinary leaf and reports
 *       its relevance in `base_fact_relevance` exactly like every constant base fact; this
 *       class then takes that one value, passes it through sigmoid unchanged (pass-through,
 *       per Phase 2 Mission 0's own Sigmoid Design Decision precedent -- a monotonic bijective
 *       single-input nonlinearity does not get its own gradient-shaped rule), and feeds it
 *       into the predicate's real, unmodified `LinearModule::propagate_relevance()` to get
 *       `relevance_wrt_x` -- a genuine end-to-end trace from a Datalog query back to the
 *       neural predicate's raw input, composed from two already-proven-correct rules rather
 *       than a new one invented for this seam.
 */
struct NeuralPredicateRelevanceResult {
    /** @brief Relevance at every extensional/base fact in the diamond program, per
     *         `RelevanceResult::base_facts` (datalog_lrp.hpp) -- includes `edge(a,b)`'s own
     *         "Datalog-level" relevance (i.e. relevance w.r.t. the neural predicate's
     *         sigmoid output `s`) before it is further propagated into the predicate's own
     *         `Module` chain below. */
    RelevanceMap base_fact_relevance;
    /** @brief `edge(a,b)`'s relevance, continued through the predicate's own sigmoid
     *         (pass-through) and `LinearModule::propagate_relevance()`, shape (1, 1) --
     *         relevance at the predicate's raw input `x`. */
    Tensor relevance_wrt_x;
};

/**
 * @brief Bridges one neural-predicate-weighted base fact into the real-valued weighted
 *        Datalog engine.
 *
 * @note **Toy program (Stage 3 design decision 3)**: the diamond-graph `edge`/`ancestor`
 *       program Mission 1 already hand-derived (`WeightedDiamondKBTest`, `datalog_weighted_engine_test.cpp`),
 *       with `edge(a,b)`'s weight replaced by a real neural predicate's output instead of the
 *       constant `0.5`:
 *       ```
 *       edge(a,b) = predicate(x)   (neural -- LinearModule(1,1) + sigmoid, this class)
 *       edge(a,c) = 0.4            (constant)
 *       edge(b,d) = 0.6            (constant)
 *       edge(c,d) = 0.3            (constant)
 *       ```
 *       with the same `ancestor(X,Y) :- edge(X,Y).` / `ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).`
 *       program. `ancestor(a,d)` is reachable via exactly two substitutions for `Z`
 *       (`Z=b`, `Z=c`), so `ancestor(a,d) = edge(a,b)*edge(b,d) + edge(a,c)*edge(c,d)
 *       = 0.6*s + 0.12`, `s` the neural predicate's output -- this both reuses a
 *       Mission-0/1-precedented, already-hand-verified toy KB shape (rather than Phase 1's
 *       `A(x)`/`B(x)` implication example, a structurally different program) and gives a
 *       genuine `⊕`-composition case (one summand is neural, the other is a pure constant),
 *       not just a single-path multiplication chain.
 * @note **Tensor-valued-weight entry (Stage 3 design decision 1)**: `WeightedFactDatabase<T>`
 *       stays scalar (`double`, and `DualNumber<double>` for the differentiable pass) -- `T`
 *       does *not* become `Tensor`. Checked directly against `tensor.hpp`'s actual public
 *       interface (recon, not assumed) before deciding: `Tensor` has no `.item()`/scalar-readout
 *       method, but does expose `at(std::initializer_list<int64_t>)` for element access, so a
 *       (1,1)-shaped `Tensor`'s single value is readable as a raw `float` via `at({0, 0})`
 *       without needing a new `Tensor` API. Making `Tensor` itself a semiring `Value` would
 *       additionally require an out-of-place elementwise `operator*` (`Tensor` only has
 *       in-place `accumulate()`, i.e. `+=`, and no multiply at all -- see tensor.hpp) -- adding
 *       one would mean touching tensor.hpp/tensor.cpp, which this mission is explicitly
 *       forbidden from doing. The neural predicate's `Tensor` output is therefore read out as
 *       a raw scalar (`float`, widened to `double`) before insertion into the fact database;
 *       the gradient is computed separately (design decision 2) and re-threaded into the
 *       predicate's own `Tensor`-shaped parameter gradients via its real `backward()`.
 * @note **Differentiability source (Stage 3 design decision 2)**: closed-form/algebraic
 *       composition (`DualSemiring<double>`, forward-mode AD -- see datalog_dual_semiring.hpp),
 *       *not* real `ComputationGraph`/`Autograd` node-level wiring. Checked directly against
 *       `computation_graph.hpp`/`node.hpp`/`autograd.hpp` before deciding: a `Node` carries
 *       only `{op_type, shape, label}` -- no value payload at all -- so there is no way for a
 *       Datalog rule's actual numeric derivation (which specific facts combined, with what
 *       weights, via how many substitutions) to be represented as graph structure; `Autograd`
 *       is explicitly scoped to "at most one parent" per node
 *       (`autograd.hpp`'s own class-level note), whereas a Datalog rule body is a
 *       variable-arity join across every one of its atoms' candidate facts -- structurally
 *       nothing like the single-parent-per-node shape `Autograd::register_backward` assumes.
 *       Wiring Datalog derivations into `ComputationGraph`/`Autograd` would therefore not be a
 *       possible-but-harder path deliberately not taken; it is not a shape either class
 *       supports today, matching the mission file's own framing that closed-form composition
 *       "is not a compromise but the only structurally sound option" once this recon is done.
 *       This class's forward pass instead runs the ordinary `naive_evaluate_weighted<DualSemiring<double>>`
 *       evaluation (zero engine changes) to get both the query weight and its exact local
 *       derivative w.r.t. the neural predicate's output in one pass; backward() then threads
 *       that scalar derivative into the predicate's real `LinearModule::backward()` (the same
 *       function `Module::forward_traced`/`Autograd::register_backward` would call for a
 *       genuinely traced module), mirroring this codebase's own existing
 *       `ToyKnowledgeBase::backward()` precedent of hand-chaining real `Module::backward()`
 *       calls with no `ComputationGraph`/`Autograd` object involved.
 */
class NeuralPredicateDatalogBridge {
public:
    /**
     * @brief Constructs the bridge with a fresh `LinearModule(1, 1)` neural predicate,
     *        initialized to the same small, non-zero weights Phase 1 Mission 2's predicate
     *        `A` used (`W=0.6, b=0.0`) -- deliberate continuity with this campaign's own
     *        established toy-KB initialization precedent, not an arbitrary new choice.
     * @param backend Backend to compute through. Not owned; must outlive this object.
     */
    explicit NeuralPredicateDatalogBridge(DeviceBackend* backend);

    /**
     * @brief Runs the neural predicate on `x`, wires its (sigmoid-squashed) scalar output in
     *        as `edge(a,b)`'s weight, and evaluates the diamond toy program under
     *        `DualSemiring<double>` to get `ancestor(a,d)`'s weight and its exact derivative
     *        w.r.t. that neural output -- all in one pass.
     * @param x Shape (1, 1) -- this bridge's toy program has exactly one grounding.
     * @return `ancestor(a,d)`'s derived weight and `d(ancestor(a,d))/d(edge(a,b))`.
     * @throws std::invalid_argument if x is not rank 2 with shape (1, 1) -- external boundary,
     *         this class's own documented shape contract (mirrors ToyKnowledgeBase::forward's
     *         own precondition-throw convention).
     */
    [[nodiscard]] NeuralPredicateQueryResult evaluate(const Tensor& x);

    /**
     * @brief Threads `evaluate()`'s `grad_wrt_predicate_output` into the neural predicate's
     *        own `LinearModule::backward()` (via the sigmoid's own closed-form derivative,
     *        `ds/dz = s*(1-s)`), accumulating `predicate().weight_grad()`/`bias_grad()`.
     * @throws std::logic_error if called before evaluate().
     */
    void backward();

    /**
     * @brief Phase 3 Mission 3's own deliverable: propagates relevance from `ancestor(a,d)`'s
     *        derived weight back through the diamond-graph derivation circuit (via
     *        `datalog_lrp::propagate_relevance_weighted`, the hand-derived `(+, x)`-circuit
     *        LRP rule) and, since `edge(a,b)` is this bridge's own neural-predicate-weighted
     *        base fact, on through the predicate's own `Module` chain (sigmoid pass-through +
     *        `LinearModule::propagate_relevance()`) -- giving relevance at the predicate's raw
     *        input `x`, composed rather than a new rule (see `NeuralPredicateRelevanceResult`'s
     *        own doc comment for the full rationale).
     * @param relevance_seed Relevance seeded at `ancestor(a,d)`'s derived weight.
     * @param config Epsilon for both the Datalog-level split and `LinearModule`'s own
     *        epsilon rule.
     * @throws std::logic_error if called before `evaluate()`.
     */
    [[nodiscard]] NeuralPredicateRelevanceResult propagate_relevance(double relevance_seed,
                                                                       const LRPRuleConfig& config = LRPRuleConfig{});

    /** @brief The neural predicate's own `LinearModule` -- test/inspection accessor. */
    [[nodiscard]] LinearModule& predicate() { return predicate_; }

    /** @brief The diamond-graph ancestor program (`ancestor(X,Y):-edge(X,Y).` /
     *         `ancestor(X,Y):-edge(X,Z),ancestor(Z,Y).`) -- exposed for tests/finite-difference
     *         harnesses that need to re-run the raw engine directly. */
    [[nodiscard]] static std::vector<Rule> diamond_ancestor_program();

    /** @brief `edge(a,c)=0.4, edge(b,d)=0.6, edge(c,d)=0.3` -- the constant-weighted facts of
     *         the diamond toy program, exposed for tests/finite-difference harnesses. */
    [[nodiscard]] static WeightedFactDatabase<double> constant_edge_facts();

private:
    DeviceBackend* backend_;
    LinearModule predicate_;
    Tensor last_x_;
    Tensor last_s_;  // (1,1) -- cached sigmoid output, sigmoid_backward's own "y" input.
    double last_grad_wrt_s_ = 0.0;
    bool has_evaluated_ = false;
};

}  // namespace pulsatrix::datalog
