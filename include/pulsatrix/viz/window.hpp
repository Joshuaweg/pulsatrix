/** @file window.hpp
 *  @brief GLFW + OpenGL3 + Dear ImGui/ImPlot window lifecycle, factored out of every demo app.
 *  @ingroup visualization
 */
#pragma once

#include <functional>
#include <string>

// Only compiled when PULSATRIX_ENABLE_VIZ builds pulsatrix_viz (see root CMakeLists.txt) --
// this is the one header in the viz module allowed to assume ImGui/ImPlot/GLFW are present.
// pulsatrix_core never includes this header or links pulsatrix_viz (charter non-negotiable
// #2's plugin-boundary discipline, extended to the native viz module).
struct GLFWwindow;

namespace pulsatrix {

/**
 * @brief Owns a GLFW window, OpenGL3 context, and ImGui/ImPlot context for the lifetime of
 *        one demo app. Every examples/viz/*_demo.cpp uses this instead of hand-rolling the
 *        standard imgui_impl_glfw_opengl3 boilerplate.
 * @note This class -- and everything that calls into ImGui or ImPlot -- is deliberately NOT
 *       unit-tested via GoogleTest (see plans/okay-we-have-now-buzzing-moth.md Testing
 *       Strategy): a live window/GL context isn't meaningfully testable headlessly on this
 *       project's dev/CI machines. Verification is via manually running the demo apps.
 */
class VizWindow {
public:
    /**
     * @brief Creates the GLFW window, OpenGL3 context, and ImGui/ImPlot contexts.
     * @throws std::runtime_error if GLFW initialization, window creation, or OpenGL loader
     *         setup fails -- external boundary: availability of a display/GPU driver is
     *         environment-dependent, not an internal invariant.
     */
    VizWindow(const std::string& title, int width, int height);

    ~VizWindow();

    VizWindow(const VizWindow&) = delete;
    VizWindow& operator=(const VizWindow&) = delete;

    /**
     * @brief Runs the main loop, calling draw_frame() once per frame between
     *        ImGui::NewFrame()/ImGui::Render(), until the window is closed.
     * @param draw_frame Caller-supplied per-frame drawing callback (ImGui/ImPlot calls only).
     */
    void run(const std::function<void()>& draw_frame);

    /**
     * @brief Same as run(draw_frame), plus after_render(framebuffer_width, framebuffer_height)
     *        called once per frame after ImGui's draw data has been rendered into the back
     *        buffer and before the buffer swap -- the one point where glReadPixels sees the
     *        finished frame (e.g. for writing screenshots).
     */
    void run(const std::function<void()>& draw_frame, const std::function<void(int, int)>& after_render);

    /** @brief Asks the main loop to exit after the current frame (e.g. a scripted/screenshot run). */
    void request_close();

private:
    GLFWwindow* window_;
};

}  // namespace pulsatrix
