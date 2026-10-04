/** @file checkpoint.hpp
 *  @brief Saving and resuming models (roadmap IO-2): parameters and buffers in one safetensors
 *         file, optimizer state in a sibling file, and a format version with a migration table.
 *  @ingroup dl_modules
 */
#pragma once

#include <map>
#include <string>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief The checkpoint format this build writes. Files record it as `format_version`; a file
 *        without one (any plain safetensors file) is version 0. Older versions load through a
 *        migration table; newer ones are rejected.
 */
inline constexpr int kCheckpointFormatVersion = 1;

/** @brief Options for LoadCheckpoint(). */
struct CheckpointLoadOptions {
    /** @brief Require the file and the model to have exactly the same names. Shapes are always
     *         checked. */
    bool strict = true;
};

/**
 * @brief Where the optimizer state for `checkpoint_path` is stored: `model.safetensors` becomes
 *        `model.optim.safetensors`; any other name gets `.optim.safetensors` appended.
 */
[[nodiscard]] std::string OptimizerStatePath(const std::string& checkpoint_path);

/**
 * @brief Writes every parameter and buffer of `model`, under their hierarchical names, to a
 *        safetensors file with `format` = "pulsatrix" and `format_version` metadata.
 * @param metadata Extra string metadata to store; `format` and `format_version` are reserved.
 * @throws std::invalid_argument if a parameter and a buffer share a name, or `metadata` uses a
 *         reserved key. @throws std::runtime_error if the file can't be written.
 */
void SaveCheckpoint(const std::string& path, Module& model, const std::map<std::string, std::string>& metadata = {});

/**
 * @brief SaveCheckpoint(), plus `optimizer`'s per-parameter state in OptimizerStatePath(path).
 * @note The optimizer file records a checksum of the model file, so a later model-only save to
 *       the same path can't be silently resumed with stale optimizer state.
 */
void SaveCheckpoint(const std::string& path, Module& model, const AdamOptimizer& optimizer,
                    const std::map<std::string, std::string>& metadata = {});

/**
 * @brief Loads parameters and buffers saved by SaveCheckpoint() (or any safetensors file whose
 *        names match) into `model`, in place.
 * @note Everything is validated before anything is written, so a failed load leaves `model`
 *       unchanged. Each tensor keeps its device and requires_grad flag.
 * @throws std::invalid_argument if the file is from a newer format version or another framework
 *         (`format` other than "pulsatrix"; importing other names is roadmap IO-4), a shape
 *         differs, an entry isn't F32, or (when strict) a name is missing or unexpected.
 * @throws std::runtime_error if the file can't be read.
 */
void LoadCheckpoint(const std::string& path, Module& model, CheckpointLoadOptions options = {});

/**
 * @brief LoadCheckpoint(), plus `optimizer`'s state from OptimizerStatePath(path), so training
 *        resumes exactly where it stopped.
 * @throws std::runtime_error if there is no optimizer file. @throws std::invalid_argument if it
 *         belongs to a different save of the model, or doesn't match the model.
 */
void LoadCheckpoint(const std::string& path, Module& model, AdamOptimizer& optimizer,
                    CheckpointLoadOptions options = {});

}  // namespace pulsatrix
