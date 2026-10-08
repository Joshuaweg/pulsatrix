#include "pulsatrix/protein_structure.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace pulsatrix {

std::optional<Point3> StructureResidue::atom(const std::string& atom_name) const {
    const auto it = atoms.find(atom_name);
    if (it == atoms.end()) return std::nullopt;
    return it->second;
}

std::string StructureChain::sequence() const {
    std::string s;
    s.reserve(residues.size());
    for (const StructureResidue& r : residues) s.push_back(r.code);
    return s;
}

const StructureChain& ProteinStructure::chain(std::string_view id) const {
    for (const StructureChain& c : chains) {
        if (c.id == id) return c;
    }
    throw std::invalid_argument("ProteinStructure: no chain \"" + std::string(id) + "\"");
}

namespace {

char OneLetter(const std::string& name) {
    static const std::unordered_map<std::string, char> codes = {
        {"ALA", 'A'}, {"ARG", 'R'}, {"ASN", 'N'}, {"ASP", 'D'}, {"CYS", 'C'}, {"GLN", 'Q'}, {"GLU", 'E'}, {"GLY", 'G'},
        {"HIS", 'H'}, {"ILE", 'I'}, {"LEU", 'L'}, {"LYS", 'K'}, {"MET", 'M'}, {"PHE", 'F'}, {"PRO", 'P'}, {"SER", 'S'},
        {"THR", 'T'}, {"TRP", 'W'}, {"TYR", 'Y'}, {"VAL", 'V'}, {"MSE", 'M'}, {"SEC", 'U'}, {"PYL", 'O'}};
    const auto it = codes.find(name);
    return it == codes.end() ? 'X' : it->second;
}

/** @brief One atom record, from either format. */
struct AtomRecord {
    bool hetero = false;
    std::string atom_name, residue_name, chain;
    int64_t number = 0;
    char insertion_code = ' ';
    Point3 position;
};

/** @brief Groups atom records into chains and residues, keeping protein residues only. */
class StructureBuilder {
public:
    void add(const AtomRecord& a) {
        if (a.hetero && a.residue_name != "MSE") return;
        if (chains_.empty() || chains_.back().id != a.chain) {
            // A chain id seen before (after a TER and ligands, say) continues that chain.
            current_ = nullptr;
            for (StructureChain& c : chains_) {
                if (c.id == a.chain) current_ = &c;
            }
            if (current_ == nullptr) {
                chains_.push_back({a.chain, {}});
                current_ = &chains_.back();
            }
        } else {
            current_ = &chains_.back();
        }
        std::vector<StructureResidue>& rs = current_->residues;
        if (rs.empty() || rs.back().number != a.number || rs.back().insertion_code != a.insertion_code) {
            rs.push_back({a.residue_name, OneLetter(a.residue_name), a.number, a.insertion_code, {}});
        }
        rs.back().atoms.emplace(a.atom_name, a.position);  // the first alternate location wins
    }

    ProteinStructure finish(const char* who) {
        ProteinStructure s;
        for (StructureChain& c : chains_) {
            std::vector<StructureResidue> kept;
            for (StructureResidue& r : c.residues) {
                if (r.atoms.count("CA") != 0) kept.push_back(std::move(r));
            }
            if (!kept.empty()) s.chains.push_back({c.id, std::move(kept)});
        }
        if (s.chains.empty()) throw std::invalid_argument(std::string(who) + ": no protein residues");
        return s;
    }

private:
    std::vector<StructureChain> chains_;
    StructureChain* current_ = nullptr;
};

std::string Trim(std::string_view s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return "";
    const size_t e = s.find_last_not_of(" \t\r");
    return std::string(s.substr(b, e - b + 1));
}

double Number(const std::string& text, const char* who, const char* what) {
    try {
        size_t used = 0;
        const double v = std::stod(text, &used);
        if (used == text.size()) return v;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument(std::string(who) + ": " + what + " \"" + text + "\" isn't a number");
}

}  // namespace

ProteinStructure ParsePdb(std::string_view text) {
    StructureBuilder builder;
    std::istringstream in{std::string(text)};
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("ENDMDL", 0) == 0) break;  // the first model only
        const bool atom = line.rfind("ATOM  ", 0) == 0, hetero = line.rfind("HETATM", 0) == 0;
        if (!atom && !hetero) continue;
        line.resize(std::max<size_t>(line.size(), 80), ' ');
        AtomRecord a;
        a.hetero = hetero;
        a.atom_name = Trim(line.substr(12, 4));
        a.residue_name = Trim(line.substr(17, 3));
        a.chain = Trim(line.substr(21, 1));
        a.number = static_cast<int64_t>(Number(Trim(line.substr(22, 4)), "ParsePdb", "residue number"));
        a.insertion_code = line[26];
        a.position = {Number(Trim(line.substr(30, 8)), "ParsePdb", "x"), Number(Trim(line.substr(38, 8)), "ParsePdb", "y"),
                      Number(Trim(line.substr(46, 8)), "ParsePdb", "z")};
        builder.add(a);
    }
    return builder.finish("ParsePdb");
}

namespace {

struct CifToken {
    std::string text;
    bool quoted = false;
};

/** @brief CIF 1.1 tokens: bare words, '...' and "..." strings, and ;-delimited text fields. */
std::vector<CifToken> CifTokens(std::string_view text) {
    std::vector<CifToken> out;
    size_t pos = 0;
    const size_t n = text.size();
    while (pos < n) {
        const size_t eol = std::min(text.find('\n', pos), n);
        std::string_view line = text.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (!line.empty() && line.front() == ';') {
            // A text field runs to the next line starting with ';'.
            std::string field(line.substr(1));
            size_t p = eol + 1;
            bool closed = false;
            while (p < n) {
                const size_t e = std::min(text.find('\n', p), n);
                std::string_view l = text.substr(p, e - p);
                if (!l.empty() && l.back() == '\r') l.remove_suffix(1);
                p = e + 1;
                if (!l.empty() && l.front() == ';') {
                    closed = true;
                    break;
                }
                field += "\n";
                field += l;
            }
            if (!closed) throw std::invalid_argument("ParseMmcif: a ;-delimited text field isn't closed");
            out.push_back({field, true});
            pos = p;
            continue;
        }
        size_t i = 0;
        while (i < line.size()) {
            if (line[i] == ' ' || line[i] == '\t') {
                ++i;
                continue;
            }
            if (line[i] == '#') break;
            if (line[i] == '\'' || line[i] == '"') {
                // The string ends at the same quote followed by whitespace or the end of the line.
                const char q = line[i];
                size_t j = i + 1;
                while (j < line.size() && !(line[j] == q && (j + 1 == line.size() || line[j + 1] == ' ' || line[j + 1] == '\t'))) ++j;
                if (j >= line.size()) throw std::invalid_argument("ParseMmcif: a quoted value isn't closed");
                out.push_back({std::string(line.substr(i + 1, j - i - 1)), true});
                i = j + 1;
                continue;
            }
            size_t j = i;
            while (j < line.size() && line[j] != ' ' && line[j] != '\t') ++j;
            out.push_back({std::string(line.substr(i, j - i)), false});
            i = j;
        }
        pos = eol + 1;
    }
    return out;
}

bool IsTag(const CifToken& t) { return !t.quoted && !t.text.empty() && t.text.front() == '_'; }

bool IsKeyword(const CifToken& t) {
    if (t.quoted) return false;
    const std::string& s = t.text;
    return s == "loop_" || s == "stop_" || s == "global_" || s.rfind("data_", 0) == 0 || s.rfind("save_", 0) == 0;
}

}  // namespace

ProteinStructure ParseMmcif(std::string_view text) {
    const std::vector<CifToken> tokens = CifTokens(text);
    std::vector<std::string> tags;
    size_t values_begin = 0, values_end = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].quoted || tokens[i].text != "loop_") continue;
        size_t j = i + 1;
        std::vector<std::string> t;
        while (j < tokens.size() && IsTag(tokens[j])) t.push_back(tokens[j++].text);
        size_t k = j;
        while (k < tokens.size() && !IsTag(tokens[k]) && !IsKeyword(tokens[k])) ++k;
        if (!t.empty() && t.front().rfind("_atom_site.", 0) == 0) {
            tags = std::move(t);
            values_begin = j;
            values_end = k;
            break;
        }
        i = k - 1;
    }
    if (tags.empty()) throw std::invalid_argument("ParseMmcif: no _atom_site loop");
    if ((values_end - values_begin) % tags.size() != 0) {
        throw std::invalid_argument("ParseMmcif: the _atom_site loop's values don't fill whole rows");
    }
    auto column = [&](std::initializer_list<const char*> names, bool required) -> std::optional<size_t> {
        for (const char* name : names) {
            for (size_t c = 0; c < tags.size(); ++c) {
                if (tags[c] == std::string("_atom_site.") + name) return c;
            }
        }
        if (required) throw std::invalid_argument(std::string("ParseMmcif: the _atom_site loop has no ") + *names.begin() + " column");
        return std::nullopt;
    };
    const size_t group = *column({"group_PDB"}, true), atom = *column({"auth_atom_id", "label_atom_id"}, true),
                 residue = *column({"auth_comp_id", "label_comp_id"}, true), chain = *column({"auth_asym_id", "label_asym_id"}, true),
                 number = *column({"auth_seq_id", "label_seq_id"}, true), x = *column({"Cartn_x"}, true), y = *column({"Cartn_y"}, true),
                 z = *column({"Cartn_z"}, true);
    const std::optional<size_t> insertion = column({"pdbx_PDB_ins_code"}, false), model = column({"pdbx_PDB_model_num"}, false);

    StructureBuilder builder;
    const size_t width = tags.size();
    std::optional<std::string> first_model;
    for (size_t r = values_begin; r < values_end; r += width) {
        auto value = [&](size_t c) -> const std::string& { return tokens[r + c].text; };
        if (model) {
            if (!first_model) first_model = value(*model);
            if (value(*model) != *first_model) break;  // the first model only
        }
        AtomRecord a;
        a.hetero = value(group) == "HETATM";
        if (!a.hetero && value(group) != "ATOM") continue;
        a.atom_name = value(atom);
        a.residue_name = value(residue);
        a.chain = value(chain);
        a.number = static_cast<int64_t>(Number(value(number), "ParseMmcif", "residue number"));
        if (insertion && value(*insertion) != "?" && value(*insertion) != ".") a.insertion_code = value(*insertion).front();
        a.position = {Number(value(x), "ParseMmcif", "x"), Number(value(y), "ParseMmcif", "y"), Number(value(z), "ParseMmcif", "z")};
        builder.add(a);
    }
    return builder.finish("ParseMmcif");
}

ProteinStructure ReadStructure(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ReadStructure: can't read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    std::string ext = path.substr(path.find_last_of('.') == std::string::npos ? path.size() : path.find_last_of('.'));
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext == ".cif" || ext == ".mmcif" ? ParseMmcif(text.str()) : ParsePdb(text.str());
}

namespace {

Point3 Sub(Point3 a, Point3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Point3 Cross(Point3 a, Point3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Point3 Unit(Point3 a) {
    const double n = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return {a.x / n, a.y / n, a.z / n};
}

/** @brief The point at @p length from @p c, at angle @p angle to b-c and dihedral @p dihedral
 *         with a (trRosetta's and ESM's `extend`). */
Point3 Extend(Point3 a, Point3 b, Point3 c, double length, double angle, double dihedral) {
    const Point3 bc = Unit(Sub(b, c));
    const Point3 n = Unit(Cross(Sub(b, a), bc));
    const Point3 m = Cross(n, bc);
    const double d0 = length * std::cos(angle), d1 = length * std::sin(angle) * std::cos(dihedral),
                 d2 = -length * std::sin(angle) * std::sin(dihedral);
    return {c.x + bc.x * d0 + m.x * d1 + n.x * d2, c.y + bc.y * d0 + m.y * d1 + n.y * d2, c.z + bc.z * d0 + m.z * d1 + n.z * d2};
}

std::optional<Point3> Representative(const StructureResidue& r, ContactAtom atom) {
    switch (atom) {
        case ContactAtom::Alpha:
            return r.atom("CA");
        case ContactAtom::Beta:
            return r.code == 'G' ? r.atom("CA") : r.atom("CB");
        case ContactAtom::VirtualBeta: {
            const auto n = r.atom("N"), ca = r.atom("CA"), c = r.atom("C");
            if (!n || !ca || !c) return std::nullopt;
            return Extend(*c, *n, *ca, 1.522, 1.927, -2.143);
        }
    }
    return std::nullopt;
}

}  // namespace

ContactMap ResidueDistances(const StructureChain& chain, ContactAtom atom) {
    const auto L = static_cast<int64_t>(chain.residues.size());
    std::vector<std::optional<Point3>> p;
    p.reserve(static_cast<size_t>(L));
    for (const StructureResidue& r : chain.residues) p.push_back(Representative(r, atom));
    ContactMap d{L, std::vector<float>(static_cast<size_t>(L * L), std::numeric_limits<float>::quiet_NaN())};
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = 0; j < L; ++j) {
            const auto& a = p[static_cast<size_t>(i)];
            const auto& b = p[static_cast<size_t>(j)];
            if (!a || !b) continue;
            const Point3 v = Sub(*a, *b);
            d.values[static_cast<size_t>(i * L + j)] = static_cast<float>(std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z));
        }
    }
    return d;
}

ContactMap TrueContacts(const StructureChain& chain, double threshold, ContactAtom atom) {
    ContactMap c = ResidueDistances(chain, atom);
    for (float& v : c.values) {
        if (!std::isnan(v)) v = v < threshold ? 1.0f : 0.0f;
    }
    return c;
}

}  // namespace pulsatrix
