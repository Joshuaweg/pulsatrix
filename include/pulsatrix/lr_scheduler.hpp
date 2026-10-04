/** @file lr_scheduler.hpp
 *  @brief Learning-rate schedules (warmup, constant, linear, cosine) and a scheduler that applies
 *         one to an optimizer and its parameter groups (roadmap TRN-4).
 *  @ingroup dl_modules
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix {

/**
 * @brief A learning-rate multiplier as a function of the step, with the formulas of Hugging Face
 *        transformers' get_{constant,linear,cosine}_schedule_with_warmup.
 * @note Warmup ramps linearly from 0 at step 0 to 1 at step `warmup`. Linear then decays to 0 at
 *       step `total`; cosine follows half a cosine down to `min_ratio` at `total`. Both stay at
 *       their final value past `total`.
 */
class LRSchedule {
public:
    /** @brief 1 after an optional linear warmup. @throws std::invalid_argument if warmup < 0. */
    [[nodiscard]] static LRSchedule Constant(int64_t warmup = 0) {
        if (warmup < 0) {
            throw std::invalid_argument("LRSchedule::Constant: warmup must be >= 0");
        }
        return LRSchedule(Kind::Constant, warmup, 0, 0.0f);
    }

    /** @brief Warmup, then linear decay to 0 at `total`. @throws std::invalid_argument unless
     *         0 <= warmup <= total and total > 0. */
    [[nodiscard]] static LRSchedule Linear(int64_t warmup, int64_t total) {
        check(warmup, total, "LRSchedule::Linear");
        return LRSchedule(Kind::Linear, warmup, total, 0.0f);
    }

    /** @brief Warmup, then cosine decay to `min_ratio` at `total`. @throws std::invalid_argument
     *         unless 0 <= warmup <= total, total > 0 and min_ratio is in [0, 1]. */
    [[nodiscard]] static LRSchedule Cosine(int64_t warmup, int64_t total, float min_ratio = 0.0f) {
        check(warmup, total, "LRSchedule::Cosine");
        if (!(min_ratio >= 0.0f && min_ratio <= 1.0f)) {
            throw std::invalid_argument("LRSchedule::Cosine: min_ratio must be in [0, 1]");
        }
        return LRSchedule(Kind::Cosine, warmup, total, min_ratio);
    }

    /** @brief The multiplier at `step`. @throws std::invalid_argument if step < 0. */
    [[nodiscard]] float multiplier(int64_t step) const {
        if (step < 0) {
            throw std::invalid_argument("LRSchedule::multiplier: step must be >= 0");
        }
        if (step < warmup_) {
            return static_cast<float>(static_cast<double>(step) / static_cast<double>(warmup_));
        }
        const double span = static_cast<double>(std::max<int64_t>(1, total_ - warmup_));
        switch (kind_) {
            case Kind::Constant:
                return 1.0f;
            case Kind::Linear:
                return static_cast<float>(std::max(0.0, static_cast<double>(total_ - step) / span));
            case Kind::Cosine: {
                constexpr double kPi = 3.14159265358979323846;
                const double progress = std::min(1.0, static_cast<double>(step - warmup_) / span);
                return static_cast<float>(min_ratio_ + (1.0 - min_ratio_) * 0.5 * (1.0 + std::cos(kPi * progress)));
            }
        }
        return 1.0f;
    }

private:
    enum class Kind { Constant, Linear, Cosine };

    LRSchedule(Kind kind, int64_t warmup, int64_t total, float min_ratio)
        : kind_(kind), warmup_(warmup), total_(total), min_ratio_(min_ratio) {}

    static void check(int64_t warmup, int64_t total, const char* who) {
        if (total <= 0 || warmup < 0 || warmup > total) {
            throw std::invalid_argument(std::string(who) + ": need 0 <= warmup <= total and total > 0");
        }
    }

    Kind kind_;
    int64_t warmup_;
    int64_t total_;
    double min_ratio_;
};

/**
 * @brief Applies an LRSchedule to an optimizer (SGDOptimizer, AdamOptimizer, AdamWOptimizer):
 *        each rate becomes its base rate times the schedule's multiplier, like PyTorch's LambdaLR.
 * @note The base rates are the optimizer's learning rate and each parameter group's (TRN-1) when
 *       the scheduler is built, so set up the groups first. Building the scheduler applies step 0
 *       (with warmup, a rate of 0). Call step() once after each optimizer step.
 */
template <typename Optimizer>
class LRScheduler {
public:
    LRScheduler(Optimizer& optimizer, LRSchedule schedule)
        : optimizer_(optimizer), schedule_(schedule), base_(optimizer.learning_rate()) {
        for (const auto& group : optimizer.param_groups()) {
            group_bases_.push_back(group.learning_rate);
        }
        apply();
    }

    /** @brief Advances one step and sets the new rates. */
    void step() {
        ++step_;
        apply();
    }

    /** @brief The step the current rates belong to. */
    [[nodiscard]] int64_t last_step() const { return step_; }

    /** @brief Jumps to `step` (e.g. when resuming from a checkpoint) and sets its rates.
     *  @throws std::invalid_argument if step < 0. */
    void set_last_step(int64_t step) {
        if (step < 0) {
            throw std::invalid_argument("LRScheduler::set_last_step: step must be >= 0");
        }
        step_ = step;
        apply();
    }

private:
    void apply() {
        auto& groups = optimizer_.param_groups();
        if (groups.size() != group_bases_.size()) {
            throw std::logic_error("LRScheduler: the optimizer's parameter groups changed after the scheduler was built");
        }
        const float m = schedule_.multiplier(step_);
        optimizer_.set_learning_rate(base_ * m);
        for (size_t i = 0; i < groups.size(); ++i) {
            groups[i].learning_rate = group_bases_[i] * m;
        }
    }

    Optimizer& optimizer_;
    LRSchedule schedule_;
    float base_;
    std::vector<float> group_bases_;
    int64_t step_ = 0;
};

}  // namespace pulsatrix
