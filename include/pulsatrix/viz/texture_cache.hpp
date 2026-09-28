/** @file texture_cache.hpp
 *  @brief Keyed GPU texture cache for uploading decoded image Tensors to OpenGL.
 *  @ingroup visualization
 */
#pragma once

#include <cstdint>
#include <unordered_map>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Opaque GL texture handle, typed to avoid a GL header include in this public header. */
using TextureId = unsigned int;

/**
 * @brief Uploads decoded image Tensors to OpenGL textures, keyed and cached by an
 *        arbitrary integer key (e.g. a Dataset index) so a scrolling image grid doesn't
 *        re-upload the same image every frame.
 * @note All real pixel-format conversion logic (Tensor -> interleaved RGB bytes) lives in
 *       plot_data.hpp's ToRgbImageBuffer, which is unit-tested; this class's only job is the
 *       GL-specific glGenTextures/glTexImage2D calls, which are not meaningfully unit-testable
 *       -- see plans/okay-we-have-now-buzzing-moth.md Testing Strategy.
 */
class TextureCache {
public:
    TextureCache() = default;
    ~TextureCache();

    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    /**
     * @brief Returns the cached texture for key, uploading image_chw first if not cached.
     * @param key Cache key (e.g. a Dataset sample index).
     * @param image_chw A rank-3 (C,H,W) or batch-1 rank-4 (1,C,H,W) image tensor -- see
     *        ToRgbImageBuffer's shape contract.
     */
    [[nodiscard]] TextureId GetOrUpload(int64_t key, const Tensor& image_chw);

    /** @brief Deletes every cached GL texture and clears the cache. */
    void Clear();

private:
    std::unordered_map<int64_t, TextureId> textures_;
};

}  // namespace pulsatrix
