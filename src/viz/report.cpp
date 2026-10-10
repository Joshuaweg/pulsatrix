#include "pulsatrix/viz/report.hpp"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "html_detail.hpp"

namespace pulsatrix {

namespace {

using Obj = JsonValue::Object;
using Arr = JsonValue::Array;

/** @brief HTML text and attribute escaping that keeps newlines and tabs (the SVG escaper drops
 *         control characters, which an SVG text node can't hold). */
std::string Escape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&#39;"; break;
            default: out += c;
        }
    }
    return out;
}

/** @brief nbformat's multiline string: a list of lines, each but the last ending in "\n". */
JsonValue Lines(const std::string& text) {
    Arr lines;
    size_t start = 0;
    while (start < text.size()) {
        const size_t nl = text.find('\n', start);
        if (nl == std::string::npos) {
            lines.emplace_back(text.substr(start));
            break;
        }
        lines.emplace_back(text.substr(start, nl - start + 1));
        start = nl + 1;
    }
    return lines;
}

bool IsJsonMime(const std::string& mime) { return mime.size() > 5 && mime.compare(mime.size() - 5, 5, "+json") == 0; }

JsonValue OutputData(const MimeBundle& bundle) {
    Obj data;
    for (const auto& [mime, value] : bundle.entries) {
        // Text formats as line lists (nbformat's convention); +json formats and base64 PNG as they are.
        if (!IsJsonMime(mime) && value.type() == JsonValue::Type::String && mime != "image/png") {
            data.emplace_back(mime, Lines(value.as_string()));
        } else {
            data.emplace_back(mime, value);
        }
    }
    return data;
}

// ---- Markdown subset ---------------------------------------------------------------------------

bool SafeUrl(const std::string& url) {
    const size_t colon = url.find(':');
    const size_t delimiter = url.find_first_of("/?#");
    if (colon == std::string::npos || (delimiter != std::string::npos && delimiter < colon)) return true;  // relative
    std::string scheme = url.substr(0, colon);
    for (char& c : scheme) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return scheme == "http" || scheme == "https" || scheme == "mailto";
}

/** @brief Inline markup: code spans first (their contents are literal), then links, bold, italic. */
std::string Inline(const std::string& text) {
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '`') {
            const size_t end = text.find('`', i + 1);
            if (end != std::string::npos) {
                out += "<code>" + Escape(text.substr(i + 1, end - i - 1)) + "</code>";
                i = end + 1;
                continue;
            }
        } else if (c == '[') {
            const size_t close = text.find("](", i + 1);
            // The URL ends at the ')' that balances its '(' , so "f(x)" can be part of it.
            size_t end = std::string::npos;
            if (close != std::string::npos) {
                int depth = 0;
                for (size_t k = close + 2; k < text.size(); ++k) {
                    if (text[k] == '(') {
                        ++depth;
                    } else if (text[k] == ')' && depth-- == 0) {
                        end = k;
                        break;
                    }
                }
            }
            if (end != std::string::npos && text.find('\n', i) > end) {
                const std::string label = text.substr(i + 1, close - i - 1), url = text.substr(close + 2, end - close - 2);
                if (SafeUrl(url)) {
                    out += "<a href=\"" + Escape(url) + "\">" + Inline(label) + "</a>";
                } else {
                    out += Inline(label);
                }
                i = end + 1;
                continue;
            }
        } else if (c == '*' && i + 1 < text.size() && text[i + 1] == '*') {
            const size_t end = text.find("**", i + 2);
            if (end != std::string::npos && end > i + 2) {
                out += "<strong>" + Inline(text.substr(i + 2, end - i - 2)) + "</strong>";
                i = end + 2;
                continue;
            }
        } else if ((c == '*' || c == '_') && i + 1 < text.size() && !std::isspace(static_cast<unsigned char>(text[i + 1]))) {
            const size_t end = text.find(c, i + 1);
            if (end != std::string::npos && end > i + 1 && !(c == '*' && end + 1 < text.size() && text[end + 1] == '*')) {
                out += "<em>" + Inline(text.substr(i + 1, end - i - 1)) + "</em>";
                i = end + 1;
                continue;
            }
        }
        out += Escape(std::string(1, c));
        ++i;
    }
    return out;
}

std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
}

std::vector<std::string> TableRow(const std::string& line) {
    std::string t = Trim(line);
    if (!t.empty() && t.front() == '|') t.erase(0, 1);
    if (!t.empty() && t.back() == '|') t.pop_back();
    std::vector<std::string> cells;
    std::string cell;
    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i] == '\\' && i + 1 < t.size() && t[i + 1] == '|') {
            cell += '|';
            ++i;
        } else if (t[i] == '|') {
            cells.push_back(Trim(cell));
            cell.clear();
        } else {
            cell += t[i];
        }
    }
    cells.push_back(Trim(cell));
    return cells;
}

bool IsTableSeparator(const std::string& line) {
    const std::string t = Trim(line);
    if (t.find('-') == std::string::npos || t.find('|') == std::string::npos) return false;
    return t.find_first_not_of("|-: ") == std::string::npos;
}

int HeadingLevel(const std::string& line) {
    int level = 0;
    while (level < static_cast<int>(line.size()) && line[static_cast<size_t>(level)] == '#') ++level;
    return level >= 1 && level <= 6 && static_cast<size_t>(level) < line.size() && line[static_cast<size_t>(level)] == ' ' ? level : 0;
}

/** @brief 0 for no list item, 1 for "- " or "* ", 2 for "1. "; sets @p rest to the item text. */
int ListItem(const std::string& line, std::string& rest) {
    const std::string t = Trim(line);
    if (t.size() > 2 && (t[0] == '-' || t[0] == '*') && t[1] == ' ') {
        rest = t.substr(2);
        return 1;
    }
    size_t d = 0;
    while (d < t.size() && std::isdigit(static_cast<unsigned char>(t[d]))) ++d;
    if (d > 0 && d + 1 < t.size() && t[d] == '.' && t[d + 1] == ' ') {
        rest = t.substr(d + 2);
        return 2;
    }
    return 0;
}

}  // namespace

std::string MarkdownToHtml(const std::string& markdown) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= markdown.size()) {
        const size_t nl = markdown.find('\n', start);
        lines.push_back(markdown.substr(start, nl == std::string::npos ? std::string::npos : nl - start));
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    std::string html;
    size_t i = 0;
    while (i < lines.size()) {
        const std::string& line = lines[i];
        const std::string t = Trim(line);
        std::string rest;
        if (t.empty()) {
            ++i;
        } else if (t.rfind("```", 0) == 0) {
            std::string code;
            for (++i; i < lines.size() && Trim(lines[i]).rfind("```", 0) != 0; ++i) code += lines[i] + "\n";
            ++i;  // the closing fence, if any
            html += "<pre><code>" + Escape(code) + "</code></pre>\n";
        } else if (const int level = HeadingLevel(t); level > 0) {
            const std::string h = "h" + std::to_string(level);
            html += "<" + h + ">" + Inline(Trim(t.substr(static_cast<size_t>(level)))) + "</" + h + ">\n";
            ++i;
        } else if (t.find('|') != std::string::npos && i + 1 < lines.size() && IsTableSeparator(lines[i + 1])) {
            html += "<table>\n<tr>";
            for (const std::string& cell : TableRow(t)) html += "<th>" + Inline(cell) + "</th>";
            html += "</tr>\n";
            for (i += 2; i < lines.size() && Trim(lines[i]).find('|') != std::string::npos; ++i) {
                html += "<tr>";
                for (const std::string& cell : TableRow(lines[i])) html += "<td>" + Inline(cell) + "</td>";
                html += "</tr>\n";
            }
            html += "</table>\n";
        } else if (const int kind = ListItem(line, rest); kind > 0) {
            const char* tag = kind == 1 ? "ul" : "ol";
            html += std::string("<") + tag + ">\n";
            while (i < lines.size() && ListItem(lines[i], rest) == kind) {
                std::string item = rest;
                // Indented lines continue the item.
                for (++i; i < lines.size() && !Trim(lines[i]).empty() && (lines[i][0] == ' ' || lines[i][0] == '\t') &&
                          ListItem(lines[i], rest) == 0;
                     ++i) {
                    item += " " + Trim(lines[i]);
                }
                html += "<li>" + Inline(item) + "</li>\n";
            }
            html += std::string("</") + tag + ">\n";
        } else {
            std::string para = t;
            for (++i; i < lines.size(); ++i) {
                const std::string next = Trim(lines[i]);
                if (next.empty() || next.rfind("```", 0) == 0 || HeadingLevel(next) > 0 || ListItem(lines[i], rest) > 0) break;
                para += "\n" + next;
            }
            html += "<p>" + Inline(para) + "</p>\n";
        }
    }
    return html;
}

// ---- Report ------------------------------------------------------------------------------------

Report::Report(std::string title) : title_(std::move(title)) {}

Report& Report::markdown(std::string text) {
    cells_.push_back({Cell::Kind::Markdown, std::move(text), false, {}});
    return *this;
}

Report& Report::code(std::string source) {
    cells_.push_back({Cell::Kind::Code, std::move(source), false, {}});
    return *this;
}

Report::Cell& Report::output_cell() {
    if (cells_.empty() || cells_.back().kind != Cell::Kind::Code) cells_.push_back({Cell::Kind::Code, "", true, {}});
    return cells_.back();
}

Report& Report::output(MimeBundle bundle) {
    if (bundle.entries.empty()) throw std::invalid_argument("Report::output: the bundle is empty");
    output_cell().outputs.push_back({false, "", std::move(bundle)});
    return *this;
}

Report& Report::text(std::string text) {
    output_cell().outputs.push_back({true, std::move(text), {}});
    return *this;
}

std::string Report::to_ipynb() const {
    Arr cells;
    int id = 0;
    auto next_id = [&id] { return "cell-" + std::to_string(++id); };
    if (!title_.empty()) {
        cells.push_back(Obj{{"cell_type", "markdown"}, {"id", next_id()}, {"metadata", Obj{}}, {"source", Lines("# " + title_)}});
    }
    for (const Cell& cell : cells_) {
        if (cell.kind == Cell::Kind::Markdown) {
            cells.push_back(Obj{{"cell_type", "markdown"}, {"id", next_id()}, {"metadata", Obj{}}, {"source", Lines(cell.source)}});
            continue;
        }
        Arr outputs;
        for (const Output& o : cell.outputs) {
            if (o.is_text) {
                outputs.push_back(Obj{{"output_type", "stream"}, {"name", "stdout"}, {"text", Lines(o.text)}});
            } else {
                outputs.push_back(Obj{{"output_type", "display_data"}, {"data", OutputData(o.bundle)}, {"metadata", Obj{}}});
            }
        }
        Obj metadata;
        if (cell.hidden_source) metadata.emplace_back("jupyter", Obj{{"source_hidden", true}});
        cells.push_back(Obj{{"cell_type", "code"},
                            {"execution_count", nullptr},
                            {"id", next_id()},
                            {"metadata", std::move(metadata)},
                            {"outputs", std::move(outputs)},
                            {"source", Lines(cell.source)}});
    }
    Obj metadata{{"kernelspec", Obj{{"display_name", "C++17"}, {"language", "cpp"}, {"name", "xcpp17"}}},
                 {"language_info", Obj{{"codemirror_mode", "text/x-c++src"},
                                       {"file_extension", ".cpp"},
                                       {"mimetype", "text/x-c++src"},
                                       {"name", "c++"},
                                       {"version", "17"}}}};
    if (!title_.empty()) metadata.emplace_back("title", title_);
    return WriteJson(Obj{{"cells", std::move(cells)}, {"metadata", std::move(metadata)}, {"nbformat", 4}, {"nbformat_minor", 5}});
}

std::string Report::to_html(const HtmlOptions& options) const {
    std::string body;
    if (!title_.empty()) body += "<h1>" + Escape(title_) + "</h1>\n";
    int charts = 0;
    for (const Cell& cell : cells_) {
        if (cell.kind == Cell::Kind::Markdown) {
            body += "<div class=\"md\">\n" + MarkdownToHtml(cell.source) + "</div>\n";
            continue;
        }
        if (!cell.hidden_source && !cell.source.empty()) body += "<pre class=\"code\"><code>" + Escape(cell.source) + "</code></pre>\n";
        for (const Output& o : cell.outputs) {
            if (o.is_text) {
                body += "<pre class=\"out\">" + Escape(o.text) + "</pre>\n";
                continue;
            }
            const MimeBundle& b = o.bundle;
            if (const JsonValue* spec = b.find(kVegaLiteMimeType)) {
                // The page loads Vega-Lite 6, which every pulsatrix spec is valid for (NB-1).
                JsonValue::Object v6 = spec->as_object();
                for (auto& [key, value] : v6) {
                    if (key == "$schema") value = "https://vega.github.io/schema/vega-lite/v6.json";
                }
                const std::string id = "vis" + std::to_string(++charts);
                body += "<figure><div id=\"" + id + "\"></div><script type=\"application/json\" id=\"" + id + "-spec\">" +
                        html_detail::ScriptSafe(WriteJson(v6)) + "</script></figure>\n";
            } else if (const JsonValue* svg = b.find("image/svg+xml")) {
                const std::string& s = svg->as_string();
                const size_t at = s.find("<svg");
                body += "<figure>" + (at == std::string::npos ? Escape(s) : s.substr(at)) + "</figure>\n";
            } else if (const JsonValue* png = b.find("image/png")) {
                body += "<figure><img alt=\"\" src=\"data:image/png;base64," + png->as_string() + "\"></figure>\n";
            } else if (const JsonValue* fragment = b.find("text/html")) {
                body += "<div class=\"out-html\">" + fragment->as_string() + "</div>\n";
            } else if (const JsonValue* plain = b.find("text/plain")) {
                body += "<pre class=\"out\">" + Escape(plain->as_string()) + "</pre>\n";
            }
        }
    }
    std::string head =
        "<style>.md{line-height:1.5;max-width:860px}.md code,pre{font-family:ui-monospace,Menlo,Consolas,monospace;"
        "font-size:13px}pre.code{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:4px;padding:8px 10px;"
        "overflow-x:auto}pre.out{margin:4px 0 12px;padding:4px 10px;border-left:3px solid #ddd;overflow-x:auto}"
        "figure img{image-rendering:pixelated;max-width:100%}.md table{margin:8px 0}"
        ".md td,.md th{border-bottom:1px solid #eee}</style>\n";
    if (charts > 0) {
        head += html_detail::VegaScripts(options);
        body += "<script>for (var i = 1; i <= " + std::to_string(charts) +
                "; i++) { (function (id) { vegaEmbed('#' + id, JSON.parse(document.getElementById(id + '-spec').textContent), "
                "{actions: {export: true, source: true, compiled: false, editor: false}}).catch(function (e) { "
                "document.getElementById(id).textContent = 'Could not draw the chart: ' + e; }); })('vis' + i); }</script>\n";
    }
    return html_detail::Page(title_.empty() ? "Report" : title_, head, body);
}

namespace {
void Save(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary);
    out << content;
    if (!out) throw std::runtime_error("Report: can't write " + path);
}
}  // namespace

void Report::save_ipynb(const std::string& path) const { Save(path, to_ipynb()); }
void Report::save_html(const std::string& path, const HtmlOptions& options) const { Save(path, to_html(options)); }

}  // namespace pulsatrix
