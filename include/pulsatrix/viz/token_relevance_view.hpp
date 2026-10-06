/** @file token_relevance_view.hpp
 *  @brief The text of a `pulsatrix.token_relevance.v1` document with each piece colored by its
 *         relevance, as an ImGui widget (VIZ-6a).
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/viz/document.hpp"

namespace pulsatrix {

/**
 * @brief Draws a document's pieces as running text that wraps to the window, each scored piece on
 *        a background colored by its relevance (DivergingColormap, scaled to the largest
 *        |relevance|), as the SVG token strip does. Unscored pieces (the spaces between words) are
 *        plain text, and a newline starts a new line. Hovering a piece shows its text and score.
 * @note ImGui's default font covers Latin-1 only, so other characters (CJK, emoji, "€") show as
 *       "?" until the application loads a font with those glyphs (ImGuiIO::Fonts); the
 *       document itself holds the right text, and the SVG strip shows it.
 * @note Call from an ImGui::Begin/End window body. Not unit-tested, like the other widgets
 *       (GoogleTest can't exercise ImGui draw calls); the document builders it is meant for are
 *       (`viz/text_relevance.hpp`).
 */
class TokenRelevanceView {
public:
    /**
     * @param id ImGui id for the widget.
     * @param shared_max_abs When > 0, the color scale's end instead of this document's largest
     *        |relevance|, so several documents shown together share one scale.
     */
    static void Draw(const char* id, const TokenRelevanceDocument& doc, float shared_max_abs = -1.0f);
};

}  // namespace pulsatrix
