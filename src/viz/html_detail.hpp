// The HTML views' page shell, shared by html.cpp and protein_views.cpp. Private to src/.
#pragma once

#include <string>

#include "pulsatrix/viz/html.hpp"

namespace pulsatrix {
namespace html_detail {

/** @brief JSON for a <script> element: "<" escaped so a label holding "</script>" can't end it. */
std::string ScriptSafe(const std::string& json);
/** @brief A script file from @p dir, escaped so it can't end its <script> element early.
 *  @throws std::runtime_error if it can't be read. */
std::string ReadScript(const std::string& dir, const char* name);
/** @brief A complete page: the shared style, @p head_extra in <head>, and @p body. */
std::string Page(const std::string& heading, const std::string& head_extra, const std::string& body);
/** @brief The <script> tags that load Vega, Vega-Lite and vega-embed as @p options says. */
std::string VegaScripts(const HtmlOptions& options);

}  // namespace html_detail
}  // namespace pulsatrix
