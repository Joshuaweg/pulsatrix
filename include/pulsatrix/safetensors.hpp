/** @file safetensors.hpp
 *  @brief Native safetensors reader and writer (roadmap IO-1): the one file format pulsatrix
 *         reads and writes for weights.
 *  @ingroup dl_modules
 *  @note A safetensors file is an 8-byte little-endian header length N, N bytes of JSON, then a
 *        packed little-endian data section. It holds only numbers, so loading one can't run code,
 *        unlike a pickle (`.pt`, `.pth`, `.pkl`).
 *  @note The reader treats every file as untrusted. It accepts only what the format allows and
 *        throws std::invalid_argument for anything else: a header longer than the file or than
 *        100 MB, JSON outside the format's subset (duplicate keys, unknown fields, non-integer or
 *        negative dimensions, invalid UTF-8), a byte range past the end of the data, a range whose
 *        size isn't element count times element size (computed without overflow), and ranges
 *        that overlap, leave holes, or leave bytes at the end unindexed.
 *  @note Two deliberate differences from the reference implementation (huggingface/safetensors
 *        0.8), which accepts both: a tensor name that appears twice is rejected (the reference
 *        keeps the last, so two tools can disagree about a file's weights), and so is an unknown
 *        field in a tensor entry (the format defines exactly dtype, shape and data_offsets).
 *        Every file the reference writes passes.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Element types a safetensors file can declare. Only F32 converts to a Tensor so far. */
enum class SafetensorsDtype { Bool, U8, I8, I16, U16, I32, U32, I64, U64, F8_E4M3, F8_E5M2, F16, BF16, F32, F64 };

/** @brief One tensor's header entry. Offsets are relative to the start of the data section. */
struct SafetensorsTensorInfo {
    SafetensorsDtype dtype;
    std::vector<int64_t> shape;
    uint64_t data_begin;
    uint64_t data_end;
};

/** @brief A parsed, fully validated safetensors file held in memory. */
class SafetensorsFile {
public:
    /**
     * @brief Reads and validates a file.
     * @throws std::runtime_error if the file can't be opened or read.
     * @throws std::invalid_argument if its contents are not a valid safetensors file.
     */
    [[nodiscard]] static SafetensorsFile Read(const std::string& path);

    /**
     * @brief Validates a file's bytes.
     * @throws std::invalid_argument if they are not a valid safetensors file.
     */
    [[nodiscard]] static SafetensorsFile Parse(std::vector<uint8_t> bytes);

    /** @brief Tensor names in storage order (by data offset). */
    [[nodiscard]] const std::vector<std::string>& names() const { return names_; }

    [[nodiscard]] bool contains(const std::string& name) const { return infos_.count(name) != 0; }

    /** @brief The header entry for `name`. @throws std::invalid_argument if there is none. */
    [[nodiscard]] const SafetensorsTensorInfo& info(const std::string& name) const;

    /** @brief The `__metadata__` string map (empty if the file has none). */
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const { return metadata_; }

    /**
     * @brief The raw little-endian bytes of `name`, as {pointer, size}, valid while this object
     *        lives. For dtypes that don't convert to a Tensor yet.
     * @throws std::invalid_argument if there is no tensor called `name`.
     */
    [[nodiscard]] std::pair<const uint8_t*, size_t> bytes(const std::string& name) const;

    /**
     * @brief Copies tensor `name` into a new Tensor on `backend`'s device.
     * @throws std::invalid_argument if there is no tensor called `name`, or its dtype isn't F32
     *         (converting F16/BF16 is roadmap IO-6).
     */
    [[nodiscard]] Tensor tensor(const std::string& name, DeviceBackend* backend) const;

private:
    SafetensorsFile() = default;

    std::vector<uint8_t> bytes_;
    size_t data_start_ = 0;
    std::vector<std::string> names_;
    std::map<std::string, SafetensorsTensorInfo> infos_;
    std::map<std::string, std::string> metadata_;
};

/**
 * @brief Serializes tensors (as F32) and string metadata into safetensors bytes.
 * @param tensors Name and tensor pairs, stored in this order. Tensors on a GPU are copied back.
 * @param metadata Stored as `__metadata__`; omitted when empty.
 * @note The header is padded with spaces so the data section starts on an 8-byte boundary, as
 *       the reference implementation does.
 * @throws std::invalid_argument if a name repeats, is `__metadata__`, or (like any metadata key
 *         or value) is not valid UTF-8.
 */
[[nodiscard]] std::vector<uint8_t> SerializeSafetensors(const std::vector<std::pair<std::string, const Tensor*>>& tensors,
                                                        const std::map<std::string, std::string>& metadata = {});

/**
 * @brief SerializeSafetensors() written to `path`.
 * @throws std::runtime_error if the file can't be written; otherwise as SerializeSafetensors().
 */
void WriteSafetensors(const std::string& path, const std::vector<std::pair<std::string, const Tensor*>>& tensors,
                      const std::map<std::string, std::string>& metadata = {});

}  // namespace pulsatrix
