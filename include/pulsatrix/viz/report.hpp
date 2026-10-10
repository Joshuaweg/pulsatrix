/** @file report.hpp
 *  @brief Notebook-format reports from C++: `.ipynb` and self-contained HTML (roadmap NB-2).
 *  @ingroup visualization
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/mime_bundle.hpp"

namespace pulsatrix {

/**
 * @brief A document of markdown, code and rich outputs, written as a Jupyter notebook
 *        (nbformat 4.5) or as one self-contained HTML page. Neither needs Python or a kernel: the
 *        notebook renders on GitHub, in JupyterLab, VS Code and Quarto with its outputs already
 *        in place. The title becomes a heading cell at the top.
 *
 * ```cpp
 * Attribution a = LRP::epsilon_plus().explain(ctx, x, target, &backend);  // computed as usual
 * Report report("LRP on ResNet18");
 * report.markdown("We explain the top class with **EpsilonPlus**.")
 *       .code("Attribution a = LRP::epsilon_plus().explain(ctx, x, target, &backend);")  // shown only
 *       .show(a);                        // any type with a mime_bundle_repr() (NB-1)
 * report.save_ipynb("resnet18.ipynb");
 * report.save_html("resnet18.html");
 * ```
 *
 * - **Code is shown, not run.** A code cell holds its source as text with no execution count.
 * - **Outputs** attach to the code cell before them. An output with no code cell before it gets
 *   a code cell of its own with no source, whose input is hidden in JupyterLab.
 * - **The notebook is deterministic:** cell ids are numbered (`cell-1`, `cell-2`, ...), so the
 *   same report always writes the same bytes.
 * - **The HTML page** renders the markdown (the subset described at to_html()) with any raw HTML
 *   in it escaped, shows code in `<pre>` blocks, and shows each output in its richest format that
 *   needs no kernel: Vega-Lite charts through vega-embed, then SVG, PNG, HTML and text.
 */
class Report {
public:
    explicit Report(std::string title = "");

    /** @brief A markdown cell. */
    Report& markdown(std::string text);
    /** @brief A code cell showing @p source as C++ (not run). */
    Report& code(std::string source);
    /** @brief A rich output: attached to the last cell if it is a code cell, otherwise in a code
     *         cell of its own. @throws std::invalid_argument for an empty bundle. */
    Report& output(MimeBundle bundle);
    /** @brief output(mime_bundle_repr(value)). */
    template <class T>
    Report& show(const T& value) {
        return output(mime_bundle_repr(value));
    }
    /** @brief A plain-text output (a `stream` output on stdout in the notebook). */
    Report& text(std::string text);

    [[nodiscard]] const std::string& title() const { return title_; }
    [[nodiscard]] size_t num_cells() const { return cells_.size(); }

    /** @brief The notebook as nbformat 4.5 JSON. Its kernel spec is xeus-cpp's C++17 kernel, so
     *         the code cells run there if someone wants to. */
    [[nodiscard]] std::string to_ipynb() const;
    /**
     * @brief One HTML page. Vega-Lite charts load Vega from the CDN (pinned, with integrity
     *        hashes) or, with HtmlScripts::Inline, embed it; a report without charts has no script.
     * @note Markdown subset: ATX headings, paragraphs, `-`, `*` and `1.` lists, fenced code,
     *       pipe tables, `**bold**`, `*italic*`, `` `code` `` and `[links](url)`; links only to
     *       http, https, mailto and relative URLs. Jupyter renders the notebook's markdown in full.
     */
    [[nodiscard]] std::string to_html(const HtmlOptions& options = {}) const;

    /** @throws std::runtime_error if the file can't be written. */
    void save_ipynb(const std::string& path) const;
    /** @throws std::runtime_error if the file can't be written. */
    void save_html(const std::string& path, const HtmlOptions& options = {}) const;

private:
    struct Output {
        bool is_text = false;  ///< a stream on stdout, else a display_data bundle
        std::string text;
        MimeBundle bundle;
    };
    struct Cell {
        enum class Kind { Markdown, Code } kind;
        std::string source;
        bool hidden_source = false;
        std::vector<Output> outputs;
    };
    Cell& output_cell();

    std::string title_;
    std::vector<Cell> cells_;
};

/** @brief Markdown (the subset Report::to_html() supports) as HTML; raw HTML is escaped. */
[[nodiscard]] std::string MarkdownToHtml(const std::string& markdown);

}  // namespace pulsatrix
