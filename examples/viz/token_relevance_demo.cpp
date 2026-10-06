/** @file token_relevance_demo.cpp
 *  @brief Shows token relevance documents (VIZ-6a) in the TokenRelevanceView widget.
 *
 *  pulsatrix_explain_text MODEL_DIR "The Eiffel Tower is located in the city of" -o tokens.json
 *  pulsatrix_explain_text MODEL_DIR "The Eiffel Tower is located in the city of" --words -o words.json
 *  token_relevance_demo tokens.json words.json [--screenshot out.png]
 *
 *  Each document gets its own section; the two share one color scale, so token and word views of
 *  the same explanation compare directly. --screenshot renders a few frames, writes a PNG and exits.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/token_relevance_view.hpp"
#include "pulsatrix/viz/window.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, char** argv) {
    using namespace pulsatrix;
    std::vector<std::string> paths;
    std::string screenshot;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--screenshot" && i + 1 < argc) {
            screenshot = argv[++i];
        } else {
            paths.push_back(a);
        }
    }
    if (paths.empty()) {
        std::fprintf(stderr, "usage: token_relevance_demo DOC.json [DOC.json ...] [--screenshot OUT.png]\n");
        return 2;
    }
    std::vector<TokenRelevanceDocument> docs;
    float shared_max_abs = 0.0f;
    try {
        for (const std::string& p : paths) {
            std::ifstream in(p, std::ios::binary);
            std::stringstream text;
            text << in.rdbuf();
            docs.push_back(ParseTokenRelevanceDocument(text.str()));
            const TokenRelevanceDocument& d = docs.back();
            for (size_t i = 0; i < d.relevance.size(); ++i) {
                if (d.is_scored(i) && std::isfinite(d.relevance[i])) shared_max_abs = std::max(shared_max_abs, std::abs(d.relevance[i]));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "token_relevance_demo: %s\n", e.what());
        return 1;
    }

    VizWindow window("pulsatrix -- token relevance", 900, 120 + 110 * static_cast<int>(docs.size()));
    int frames = 0;
    window.run(
        [&]() {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin("token relevance", nullptr, ImGuiWindowFlags_NoDecoration);
            for (size_t i = 0; i < docs.size(); ++i) {
                if (i > 0) ImGui::Separator();
                TokenRelevanceView::Draw(paths[i].c_str(), docs[i], shared_max_abs);
            }
            ImGui::End();
        },
        [&](int w, int h) {
            if (screenshot.empty() || ++frames < 4) return;  // let the layout settle
            std::vector<unsigned char> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadBuffer(GL_BACK);
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
            for (size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
            stbi_flip_vertically_on_write(1);
            if (stbi_write_png(screenshot.c_str(), w, h, 4, rgba.data(), w * 4) != 0) std::printf("wrote %s\n", screenshot.c_str());
            window.request_close();
        });
    return 0;
}
