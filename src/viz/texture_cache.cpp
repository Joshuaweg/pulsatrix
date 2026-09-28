#include "pulsatrix/viz/texture_cache.hpp"

#include <GLFW/glfw3.h>

#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

TextureCache::~TextureCache() { Clear(); }

void TextureCache::Clear() {
    for (const auto& [key, id] : textures_) {
        GLuint gl_id = static_cast<GLuint>(id);
        glDeleteTextures(1, &gl_id);
    }
    textures_.clear();
}

TextureId TextureCache::GetOrUpload(int64_t key, const Tensor& image_chw) {
    auto it = textures_.find(key);
    if (it != textures_.end()) {
        return it->second;
    }

    RgbImageBuffer buffer = ToRgbImageBuffer(image_chw);

    GLuint gl_id = 0;
    glGenTextures(1, &gl_id);
    glBindTexture(GL_TEXTURE_2D, gl_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, static_cast<GLsizei>(buffer.width), static_cast<GLsizei>(buffer.height), 0,
                 GL_RGB, GL_UNSIGNED_BYTE, buffer.pixels.data());

    TextureId id = static_cast<TextureId>(gl_id);
    textures_[key] = id;
    return id;
}

}  // namespace pulsatrix
