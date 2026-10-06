#include "pulsatrix/text_tokenizer.hpp"

#include <algorithm>
#include <stdexcept>

#include "utf8.hpp"

namespace pulsatrix {

void Encoding::push_back(int64_t id, std::string token, Offset offset, bool special) {
    ids.push_back(id);
    tokens.push_back(std::move(token));
    offsets.push_back(offset);
    special_tokens_mask.push_back(special ? 1 : 0);
}

// ---- NormalizedString ----------------------------------------------------------------------

NormalizedString::NormalizedString(std::string text, size_t offset) : text_(std::move(text)) {
    origin_.reserve(text_.size());
    for (size_t i = 0; i < text_.size(); ++i) origin_.push_back({offset + i, offset + i + 1});
}

Offset NormalizedString::original(size_t begin, size_t end) const {
    if (begin > end || end > text_.size()) throw std::out_of_range("NormalizedString::original: range outside the text");
    if (begin == end) {
        if (origin_.empty()) return {};
        const size_t at = begin < origin_.size() ? origin_[begin].begin : origin_.back().end;
        return {at, at};
    }
    Offset span = origin_[begin];
    for (size_t i = begin + 1; i < end; ++i) {
        span.begin = std::min(span.begin, origin_[i].begin);
        span.end = std::max(span.end, origin_[i].end);
    }
    return span;
}

NormalizedString NormalizedString::slice(size_t begin, size_t end) const {
    if (begin > end || end > text_.size()) throw std::out_of_range("NormalizedString::slice: range outside the text");
    NormalizedString out;
    out.text_ = text_.substr(begin, end - begin);
    out.origin_.assign(origin_.begin() + static_cast<ptrdiff_t>(begin), origin_.begin() + static_cast<ptrdiff_t>(end));
    return out;
}

void NormalizedString::rebuild(const std::vector<Piece>& pieces) {
    std::string text;
    std::vector<Offset> origin;
    for (const Piece& p : pieces) {
        const Offset span = original(p.source_begin, p.source_end);
        text += p.text;
        origin.insert(origin.end(), p.text.size(), span);
    }
    text_ = std::move(text);
    origin_ = std::move(origin);
}

// ---- TextTokenizer -------------------------------------------------------------------------

TextTokenizer::TextTokenizer(std::shared_ptr<const TokenModel> model) : model_(std::move(model)) {
    if (!model_) throw std::invalid_argument("TextTokenizer: needs a model");
}

TextTokenizer& TextTokenizer::set_normalizer(std::shared_ptr<const Normalizer> normalizer) {
    normalizer_ = std::move(normalizer);
    return *this;
}

TextTokenizer& TextTokenizer::set_pre_tokenizer(std::shared_ptr<const PreTokenizer> pre_tokenizer) {
    pre_tokenizer_ = std::move(pre_tokenizer);
    return *this;
}

TextTokenizer& TextTokenizer::set_post_processor(std::shared_ptr<const PostProcessor> post_processor) {
    post_processor_ = std::move(post_processor);
    return *this;
}

TextTokenizer& TextTokenizer::set_decoder(std::shared_ptr<const Decoder> decoder) {
    decoder_ = std::move(decoder);
    return *this;
}

TextTokenizer& TextTokenizer::add_token(AddedToken token) {
    if (token.content.empty()) throw std::invalid_argument("TextTokenizer::add_token: empty content");
    if (!utf8::IsValid(token.content)) throw std::invalid_argument("TextTokenizer::add_token: content isn't valid UTF-8");
    if (added_by_content_.count(token.content) != 0) {
        throw std::invalid_argument("TextTokenizer::add_token: \"" + token.content + "\" is already added");
    }
    if (added_by_id_.count(token.id) != 0) {
        throw std::invalid_argument("TextTokenizer::add_token: id " + std::to_string(token.id) + " is already added");
    }
    const size_t index = added_.size();
    added_by_content_[token.content] = index;
    added_by_id_[token.id] = index;
    std::vector<size_t>& bucket = added_by_first_byte_[static_cast<unsigned char>(token.content[0])];
    added_.push_back(std::move(token));
    bucket.push_back(index);
    std::stable_sort(bucket.begin(), bucket.end(),
                     [&](size_t a, size_t b) { return added_[a].content.size() > added_[b].content.size(); });
    return *this;
}

void TextTokenizer::encode_segment(std::string_view text, size_t offset, Encoding& out) const {
    if (text.empty()) return;
    NormalizedString normalized{std::string(text), offset};
    if (normalizer_) normalizer_->normalize(normalized);
    std::vector<NormalizedString> splits;
    splits.push_back(std::move(normalized));
    if (pre_tokenizer_) pre_tokenizer_->pre_tokenize(splits);
    for (const NormalizedString& split : splits) {
        if (split.empty()) continue;
        for (ModelToken& t : model_->tokenize(split.text())) {
            out.push_back(t.id, std::move(t.value), split.original(t.begin, t.end), false);
        }
    }
}

Encoding TextTokenizer::encode(std::string_view text, bool add_special_tokens) const {
    if (!utf8::IsValid(text)) throw std::invalid_argument("TextTokenizer::encode: the text isn't valid UTF-8");
    Encoding out;
    size_t segment_start = 0;
    size_t i = 0;
    while (i < text.size()) {
        const AddedToken* match = nullptr;
        const auto bucket = added_by_first_byte_.find(static_cast<unsigned char>(text[i]));
        if (bucket != added_by_first_byte_.end()) {
            for (size_t index : bucket->second) {
                const std::string& content = added_[index].content;
                if (text.compare(i, content.size(), content) == 0) {
                    match = &added_[index];
                    break;  // longest first
                }
            }
        }
        if (match == nullptr) {
            ++i;
            continue;
        }
        encode_segment(text.substr(segment_start, i - segment_start), segment_start, out);
        out.push_back(match->id, match->content, {i, i + match->content.size()}, match->special);
        i += match->content.size();
        segment_start = i;
    }
    encode_segment(text.substr(segment_start), segment_start, out);
    return post_processor_ ? post_processor_->process(std::move(out), add_special_tokens) : out;
}

std::string TextTokenizer::decode(const std::vector<int64_t>& ids, bool skip_special_tokens) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int64_t id : ids) {
        const auto added = added_by_id_.find(id);
        if (added != added_by_id_.end()) {
            if (!(skip_special_tokens && added_[added->second].special)) tokens.push_back(added_[added->second].content);
            continue;
        }
        std::optional<std::string> token = model_->id_to_token(id);
        if (!token) throw std::out_of_range("TextTokenizer::decode: no token has id " + std::to_string(id));
        tokens.push_back(std::move(*token));
    }
    if (decoder_) return decoder_->decode(tokens);
    std::string text;  // no decoder: joined with spaces, as Hugging Face does
    for (size_t i = 0; i < tokens.size(); ++i) text += (i == 0 ? "" : " ") + tokens[i];
    return text;
}

int64_t TextTokenizer::vocab_size() const {
    int64_t size = model_->vocab_size();
    for (const AddedToken& t : added_) size = std::max(size, t.id + 1);
    return size;
}

std::optional<int64_t> TextTokenizer::token_to_id(std::string_view token) const {
    const auto added = added_by_content_.find(std::string(token));
    if (added != added_by_content_.end()) return added_[added->second].id;
    return model_->token_to_id(token);
}

std::optional<std::string> TextTokenizer::id_to_token(int64_t id) const {
    const auto added = added_by_id_.find(id);
    if (added != added_by_id_.end()) return added_[added->second].content;
    return model_->id_to_token(id);
}

}  // namespace pulsatrix
