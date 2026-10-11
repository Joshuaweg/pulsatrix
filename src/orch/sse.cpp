#include "pulsatrix/orch/sse.hpp"

#include <stdexcept>
#include <utility>

namespace pulsatrix::orch {

SseParser::SseParser(Handler on_event, std::size_t max_event_bytes)
    : on_event_(std::move(on_event)), max_event_bytes_(max_event_bytes) {}

void SseParser::feed(std::string_view chunk) {
    std::size_t i = 0;
    // The stream may start with a UTF-8 byte-order mark, possibly split across chunks.
    static constexpr std::string_view kBom = "\xEF\xBB\xBF";
    while (!started_ && i < chunk.size()) {
        if (chunk[i] == kBom[bom_matched_]) {
            ++i;
            started_ = ++bom_matched_ == kBom.size();
        } else {
            // Not a byte-order mark after all: the bytes held back are the start of a line.
            line_.assign(kBom.substr(0, bom_matched_));
            started_ = true;
        }
    }
    if (after_cr_ && i < chunk.size()) {
        after_cr_ = false;
        if (chunk[i] == '\n') {
            ++i;
        }
    }
    while (i < chunk.size()) {
        const std::size_t end = chunk.find_first_of("\r\n", i);
        if (end == std::string_view::npos) {
            if (line_.size() + (chunk.size() - i) > max_event_bytes_) {
                throw std::length_error("server-sent event line longer than " + std::to_string(max_event_bytes_) + " bytes");
            }
            line_.append(chunk.substr(i));
            return;
        }
        std::string_view line = chunk.substr(i, end - i);
        if (!line_.empty()) {
            line_.append(line);
            line = line_;
        }
        if (line.size() > max_event_bytes_) {
            throw std::length_error("server-sent event line longer than " + std::to_string(max_event_bytes_) + " bytes");
        }
        process_line(line);
        line_.clear();
        i = end + 1;
        if (chunk[end] == '\r') {
            if (i == chunk.size()) {
                after_cr_ = true;
            } else if (chunk[i] == '\n') {
                ++i;
            }
        }
    }
}

void SseParser::finish() {
    // An unterminated last line is not a complete event, and neither is an event without its
    // blank line, so both are dropped.
    line_.clear();
    data_.clear();
    type_.clear();
    has_data_ = false;
    after_cr_ = false;
}

void SseParser::process_line(std::string_view line) {
    if (line.empty()) {
        dispatch();
        return;
    }
    if (line.front() == ':') {
        return;  // a comment, often sent as a keep-alive
    }
    std::string_view field = line;
    std::string_view value;
    if (const std::size_t colon = line.find(':'); colon != std::string_view::npos) {
        field = line.substr(0, colon);
        value = line.substr(colon + 1);
        if (!value.empty() && value.front() == ' ') {
            value.remove_prefix(1);
        }
    }
    if (field == "data") {
        if (data_.size() + value.size() + 1 > max_event_bytes_) {
            throw std::length_error("server-sent event larger than " + std::to_string(max_event_bytes_) + " bytes");
        }
        data_.append(value);
        data_ += '\n';
        has_data_ = true;
    } else if (field == "event") {
        type_.assign(value);
    } else if (field == "id") {
        if (value.find('\0') == std::string_view::npos) {
            last_event_id_.assign(value);
        }
    } else if (field == "retry") {
        if (!value.empty() && value.find_first_not_of("0123456789") == std::string_view::npos) {
            try {
                retry_ = std::chrono::milliseconds(std::stoll(std::string(value)));
            } catch (const std::out_of_range&) {
                // Too large to be a real reconnection time: ignored like any invalid value.
            }
        }
    }
    // Any other field is ignored.
}

void SseParser::dispatch() {
    if (!has_data_) {
        type_.clear();  // the specification resets the type even when nothing is dispatched
        return;
    }
    SseEvent event;
    if (!type_.empty()) {
        event.type = std::move(type_);
    }
    data_.pop_back();  // the newline after the last data line
    event.data = std::move(data_);
    event.last_event_id = last_event_id_;
    data_.clear();
    type_.clear();
    has_data_ = false;
    on_event_(event);
}

}  // namespace pulsatrix::orch
