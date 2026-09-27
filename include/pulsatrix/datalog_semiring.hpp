/** @file datalog_semiring.hpp
 *  @brief Generic provenance-semiring abstraction (`zero`/`one`/`add`=(+)/`mul`=(x)) plus the
 *         boolean (trivial) and real-valued (+, x) concrete instantiations. Phase 3 Mission 1
 *         of campaign_exai_dl_library_neuro_symbolic (Generic Provenance-Semiring Abstraction).
 *  @ingroup dl_modules
 */
#pragma once

#include <type_traits>

namespace pulsatrix::datalog {

/**
 * @brief A "Semiring" here is not a base class -- it is a compile-time *trait shape* a type
 *        must satisfy to be usable as `naive_evaluate_weighted<Semiring>`'s template
 *        parameter: a nested `Value` type alias plus four `static` functions, `zero()`,
 *        `one()`, `add(Value,Value)` (`⊕`), `mul(Value,Value)` (`⊗`). There is no virtual
 *        interface anywhere in this file.
 *
 * @note Stage 3 design decision 3 (templating mechanism): a compile-time template parameter
 *       with a duck-typed trait shape, not a type-erased/virtual interface. Rationale, checked
 *       against this codebase's own precedent rather than assumed:
 *       - The mission's own open-design-question text guessed this would be "closer to how
 *         `DeviceBackend` is chosen" -- recon of `include/pulsatrix/device_backend.hpp` shows
 *         that guess was backwards: `DeviceBackend` is an *abstract virtual* interface, chosen
 *         at **runtime** per-`Tensor` (`DeviceType::Cpu/Cuda/Hip`), precisely because a single
 *         running program legitimately mixes backends (a `Tensor` moves between devices via
 *         `Tensor::to()`). A Datalog evaluation's semiring choice has no such runtime-mixing
 *         requirement -- one `naive_evaluate_weighted` call always uses exactly one semiring,
 *         decided at the call site, never switched mid-evaluation.
 *       - The actual matching precedent, found by checking `include/pulsatrix/selection.hpp`
 *         (the evolutionary-algorithms family), is a *duck-typed template parameter* pattern:
 *         `template <typename Genotype, typename FitnessT, typename RNG>` free functions, no
 *         C++20 `concept` (this codebase targets C++17), no shared base class -- the compiler
 *         enforces the shape at the instantiation call site via ordinary overload
 *         resolution/member lookup. `Semiring` here follows that exact precedent.
 *       - Templates give zero runtime overhead (every `add`/`mul` call is inlined at
 *         `naive_evaluate_weighted<BooleanSemiring>` / `<RealSemiring<double>>`'s own call
 *         site, no vtable indirection per fixpoint-round arithmetic operation, which matters
 *         since these run inside the evaluator's innermost loop), and they make "the boolean
 *         semiring is the trivial case, not special-cased engine logic" (campaign doc's own
 *         phrasing) literally true: `naive_evaluate_weighted`'s body contains no
 *         `if constexpr`/`if (is_boolean)` branch anywhere -- `BooleanSemiring` and
 *         `RealSemiring<double>` are two ordinary instantiations of one generic algorithm.
 */

/**
 * @brief The boolean semiring: `⊕ = OR`, `⊗ = AND`, `zero = false`, `one = true`.
 * @note This is Mission 0's implicit boolean-only engine semantics, now made an explicit,
 *       swappable instantiation rather than hardcoded evaluator logic (the mission's own
 *       framing: "boolean semiring re-expressed as the trivial case").
 */
struct BooleanSemiring {
    using Value = bool;

    [[nodiscard]] static constexpr Value zero() { return false; }
    [[nodiscard]] static constexpr Value one() { return true; }
    [[nodiscard]] static constexpr Value add(Value a, Value b) { return a || b; }
    [[nodiscard]] static constexpr Value mul(Value a, Value b) { return a && b; }
};

/**
 * @brief The real-valued `(+, x)` semiring: `⊕` = floating-point addition, `⊗` = floating-point
 *        multiplication, `zero = 0.0`, `one = 1.0` -- the differentiable-provenance case.
 * @note Operates on a plain `T` (`double`/`float`) scalar. Mission 2's exclusive scope is
 *       wiring this `T` to a `Tensor`/neural-predicate output and computing a gradient; this
 *       mission deliberately stops short of that -- `T` here is never anything but a plain
 *       arithmetic scalar.
 */
template <typename T>
struct RealSemiring {
    static_assert(std::is_floating_point_v<T>, "RealSemiring<T> requires a floating-point T (double or float)");

    using Value = T;

    [[nodiscard]] static constexpr Value zero() { return static_cast<T>(0); }
    [[nodiscard]] static constexpr Value one() { return static_cast<T>(1); }
    [[nodiscard]] static constexpr Value add(Value a, Value b) { return a + b; }
    [[nodiscard]] static constexpr Value mul(Value a, Value b) { return a * b; }
};

}  // namespace pulsatrix::datalog
