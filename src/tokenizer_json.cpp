#include "pulsatrix/tokenizer_json.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "pulsatrix/byte_level_bpe.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/sentencepiece_bpe.hpp"
#include "pulsatrix/tokenizer_components.hpp"

namespace pulsatrix {

namespace {

[[noreturn]] void Unsupported(const std::string& what) {
    throw std::invalid_argument("ParseTokenizerJson: unsupported " + what);
}

const JsonValue& Member(const JsonValue& obj, std::string_view key) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr) throw std::invalid_argument("ParseTokenizerJson: missing \"" + std::string(key) + "\"");
    return *v;
}

bool Flag(const JsonValue& obj, std::string_view key, bool fallback) {
    const JsonValue* v = obj.find(key);
    return v == nullptr || v->is_null() ? fallback : v->as_bool();
}

std::string OptionalString(const JsonValue& obj, std::string_view key) {
    const JsonValue* v = obj.find(key);
    return v == nullptr || v->is_null() ? std::string() : v->as_string();
}

const std::string& TypeOf(const JsonValue& component) { return Member(component, "type").as_string(); }

/** @brief A Replace pattern, {"String": ...} or {"Regex": ...}: (is_regex, text). */
std::pair<bool, std::string> Pattern(const JsonValue& j) {
    const JsonValue& pattern = Member(j, "pattern");
    if (const JsonValue* r = pattern.find("Regex")) return {true, r->as_string()};
    if (const JsonValue* s = pattern.find("String")) return {false, s->as_string()};
    Unsupported("pattern (neither Regex nor String)");
}

PrependScheme Scheme(const JsonValue& j) {
    if (const JsonValue* scheme = j.find("prepend_scheme"); scheme != nullptr && !scheme->is_null()) {
        const std::string& s = scheme->as_string();
        if (s == "always") return PrependScheme::Always;
        if (s == "first") return PrependScheme::First;
        if (s == "never") return PrependScheme::Never;
        Unsupported("Metaspace prepend_scheme \"" + s + "\"");
    }
    return Flag(j, "add_prefix_space", true) ? PrependScheme::Always : PrependScheme::Never;  // older files
}

std::shared_ptr<const Normalizer> MakeNormalizer(const JsonValue& j) {
    const std::string& type = TypeOf(j);
    if (type == "NFC") return std::make_shared<NfcNormalizer>();
    if (type == "Replace") {
        const auto [regex, pattern] = Pattern(j);
        const std::string& content = Member(j, "content").as_string();
        return regex ? std::make_shared<ReplaceNormalizer>(ReplaceNormalizer::Regex(pattern, content))
                     : std::make_shared<ReplaceNormalizer>(pattern, content);
    }
    if (type == "Prepend") return std::make_shared<PrependNormalizer>(Member(j, "prepend").as_string());
    if (type == "Sequence") {
        std::vector<std::shared_ptr<const Normalizer>> parts;
        for (const JsonValue& n : Member(j, "normalizers").as_array()) parts.push_back(MakeNormalizer(n));
        return std::make_shared<SequenceNormalizer>(std::move(parts));
    }
    Unsupported("normalizer \"" + type + "\"");
}

SplitBehavior Behavior(const std::string& name) {
    if (name == "Removed") return SplitBehavior::Removed;
    if (name == "Isolated") return SplitBehavior::Isolated;
    if (name == "MergedWithPrevious") return SplitBehavior::MergedWithPrevious;
    if (name == "MergedWithNext") return SplitBehavior::MergedWithNext;
    if (name == "Contiguous") return SplitBehavior::Contiguous;
    Unsupported("split behavior \"" + name + "\"");
}

std::shared_ptr<const PreTokenizer> MakePreTokenizer(const JsonValue& j) {
    const std::string& type = TypeOf(j);
    if (type == "Sequence") {
        std::vector<std::shared_ptr<const PreTokenizer>> parts;
        for (const JsonValue& p : Member(j, "pretokenizers").as_array()) parts.push_back(MakePreTokenizer(p));
        return std::make_shared<SequencePreTokenizer>(std::move(parts));
    }
    if (type == "Split") {
        const JsonValue& pattern = Member(j, "pattern");
        const SplitBehavior behavior = Behavior(Member(j, "behavior").as_string());
        const bool invert = Flag(j, "invert", false);
        if (const JsonValue* regex = pattern.find("Regex")) {
            return std::make_shared<SplitPreTokenizer>(regex->as_string(), behavior, invert);
        }
        if (const JsonValue* literal = pattern.find("String")) {
            return std::make_shared<SplitPreTokenizer>(SplitPreTokenizer::Literal(literal->as_string(), behavior, invert));
        }
        Unsupported("Split pattern (neither Regex nor String)");
    }
    if (type == "Digits") return std::make_shared<DigitsPreTokenizer>(Flag(j, "individual_digits", false));
    if (type == "Metaspace") {
        return std::make_shared<MetaspacePreTokenizer>(Member(j, "replacement").as_string(), Scheme(j), Flag(j, "split", true));
    }
    if (type == "ByteLevel") {
        return std::make_shared<ByteLevelPreTokenizer>(Flag(j, "add_prefix_space", true), Flag(j, "use_regex", true));
    }
    Unsupported("pre-tokenizer \"" + type + "\"");
}

std::shared_ptr<const TokenModel> MakeModel(const JsonValue& j) {
    const JsonValue* type_value = j.find("type");
    const std::string type = type_value != nullptr ? type_value->as_string() : "BPE";
    if (type != "BPE" && type != "WordLevel") Unsupported("model \"" + type + "\"");
    std::unordered_map<std::string, int64_t> vocab;
    for (const auto& [token, id] : Member(j, "vocab").as_object()) vocab.emplace(token, id.as_int64());
    if (type == "WordLevel") {
        std::vector<std::string> tokens(vocab.size());
        for (const auto& [token, id] : vocab) {
            if (id < 0 || static_cast<size_t>(id) >= tokens.size()) Unsupported("WordLevel vocabulary with gaps in its ids");
            tokens[static_cast<size_t>(id)] = token;
        }
        return std::make_shared<WordLevelModel>(std::move(tokens), Member(j, "unk_token").as_string());
    }
    if (type != "BPE") Unsupported("model \"" + type + "\"");
    if (const JsonValue* dropout = j.find("dropout"); dropout != nullptr && !dropout->is_null()) Unsupported("BPE dropout");
    std::vector<std::pair<std::string, std::string>> merges;
    const JsonValue::Array& raw = Member(j, "merges").as_array();
    merges.reserve(raw.size());
    for (const JsonValue& m : raw) {
        if (m.type() == JsonValue::Type::String) {
            const std::string& s = m.as_string();
            const size_t space = s.find(' ');
            if (space == std::string::npos || s.find(' ', space + 1) != std::string::npos) {
                throw std::invalid_argument("ParseTokenizerJson: merge \"" + s + "\" isn't two tokens separated by a space");
            }
            merges.emplace_back(s.substr(0, space), s.substr(space + 1));
        } else {
            const JsonValue::Array& pair = m.as_array();
            if (pair.size() != 2) throw std::invalid_argument("ParseTokenizerJson: a merge pair must hold two tokens");
            merges.emplace_back(pair[0].as_string(), pair[1].as_string());
        }
    }
    BpeOptions options;
    options.ignore_merges = Flag(j, "ignore_merges", false);
    options.continuing_subword_prefix = OptionalString(j, "continuing_subword_prefix");
    options.end_of_word_suffix = OptionalString(j, "end_of_word_suffix");
    if (const JsonValue* unk = j.find("unk_token"); unk != nullptr && !unk->is_null()) options.unk_token = unk->as_string();
    options.byte_fallback = Flag(j, "byte_fallback", false);
    options.fuse_unk = Flag(j, "fuse_unk", false);
    return std::make_shared<BpeModel>(std::move(vocab), merges, std::move(options));
}

std::shared_ptr<const PostProcessor> MakePostProcessor(const JsonValue& j) {
    const std::string& type = TypeOf(j);
    if (type == "ByteLevel") {
        return std::make_shared<ByteLevelPostProcessor>(Flag(j, "add_prefix_space", true), Flag(j, "trim_offsets", true));
    }
    if (type == "Sequence") {
        std::vector<std::shared_ptr<const PostProcessor>> parts;
        for (const JsonValue& p : Member(j, "processors").as_array()) parts.push_back(MakePostProcessor(p));
        return std::make_shared<SequencePostProcessor>(std::move(parts));
    }
    if (type == "TemplateProcessing") {
        const JsonValue& specials = Member(j, "special_tokens");
        std::vector<TemplatePostProcessor::Item> single;
        for (const JsonValue& item : Member(j, "single").as_array()) {
            if (const JsonValue* seq = item.find("Sequence")) {
                if (Member(*seq, "id").as_string() != "A") Unsupported("TemplateProcessing single template using sequence B");
                single.push_back({});
                continue;
            }
            const std::string& key = Member(Member(item, "SpecialToken"), "id").as_string();
            const JsonValue& special = Member(specials, key);
            TemplatePostProcessor::Item s;
            for (const JsonValue& id : Member(special, "ids").as_array()) s.special_ids.push_back(id.as_int64());
            for (const JsonValue& t : Member(special, "tokens").as_array()) s.special_tokens.push_back(t.as_string());
            if (s.special_ids.empty()) Unsupported("TemplateProcessing special token with no ids");
            single.push_back(std::move(s));
        }
        return std::make_shared<TemplatePostProcessor>(std::move(single));
    }
    Unsupported("post-processor \"" + type + "\"");
}

std::shared_ptr<const Decoder> MakeDecoder(const JsonValue& j) {
    const std::string& type = TypeOf(j);
    if (type == "ByteLevel") return std::make_shared<ByteLevelDecoder>();
    if (type == "Fuse") return std::make_shared<FuseDecoder>();
    if (type == "ByteFallback") return std::make_shared<ByteFallbackDecoder>();
    if (type == "Replace") {
        const auto [regex, pattern] = Pattern(j);
        const std::string& content = Member(j, "content").as_string();
        return regex ? std::make_shared<ReplaceDecoder>(ReplaceDecoder::Regex(pattern, content))
                     : std::make_shared<ReplaceDecoder>(pattern, content);
    }
    if (type == "Strip") {
        return std::make_shared<StripDecoder>(Member(j, "content").as_string(), static_cast<size_t>(Member(j, "start").as_int64()),
                                              static_cast<size_t>(Member(j, "stop").as_int64()));
    }
    if (type == "Metaspace") return std::make_shared<MetaspaceDecoder>(Member(j, "replacement").as_string(), Scheme(j));
    if (type == "Sequence") {
        std::vector<std::shared_ptr<const Decoder>> parts;
        for (const JsonValue& d : Member(j, "decoders").as_array()) parts.push_back(MakeDecoder(d));
        return std::make_shared<SequenceDecoder>(std::move(parts));
    }
    Unsupported("decoder \"" + type + "\"");
}

}  // namespace

TextTokenizer ParseTokenizerJson(std::string_view json) {
    const JsonValue root = ParseJson(json);
    TextTokenizer tokenizer(MakeModel(Member(root, "model")));
    const JsonValue* normalizer = root.find("normalizer");
    const bool has_normalizer = normalizer != nullptr && !normalizer->is_null();
    if (has_normalizer) tokenizer.set_normalizer(MakeNormalizer(*normalizer));
    if (const JsonValue* p = root.find("pre_tokenizer"); p != nullptr && !p->is_null()) tokenizer.set_pre_tokenizer(MakePreTokenizer(*p));
    if (const JsonValue* p = root.find("post_processor"); p != nullptr && !p->is_null()) tokenizer.set_post_processor(MakePostProcessor(*p));
    if (const JsonValue* d = root.find("decoder"); d != nullptr && !d->is_null()) tokenizer.set_decoder(MakeDecoder(*d));
    if (const JsonValue* added = root.find("added_tokens"); added != nullptr && !added->is_null()) {
        for (const JsonValue& t : added->as_array()) {
            const std::string& content = Member(t, "content").as_string();
            for (const char* flag : {"lstrip", "rstrip", "single_word"}) {
                if (Flag(t, flag, false)) Unsupported(std::string("added token option \"") + flag + "\" (on \"" + content + "\")");
            }
            if (Flag(t, "normalized", false) && has_normalizer) {
                Unsupported("added token matched after normalization (on \"" + content + "\")");
            }
            tokenizer.add_token({content, Member(t, "id").as_int64(), Flag(t, "special", false)});
        }
    }
    return tokenizer;
}

TextTokenizer LoadTokenizerJson(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("LoadTokenizerJson: can't read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return ParseTokenizerJson(text.str());
}

}  // namespace pulsatrix
