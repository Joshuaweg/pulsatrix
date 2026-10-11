/** @file sse.hpp
 *  @brief A parser for server-sent events, the format LLM APIs stream their responses in.
 *  @ingroup orchestrator
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace pulsatrix::orch {

/** @brief One server-sent event. */
struct SseEvent {
    /** @brief The `event:` field, or `message` when the event has none. */
    std::string type = "message";
    /** @brief The `data:` lines joined with newlines. */
    std::string data;
    /** @brief The last `id:` seen in the stream so far, as the specification defines it. */
    std::string last_event_id;
};

/**
 * @brief Parses a `text/event-stream` body fed in chunks of any size, as the WHATWG HTML
 *        specification defines it.
 * @note Lines may end in CRLF, LF or CR, also split across chunks. A leading byte-order mark,
 *       comment lines (`:`) and unknown fields are ignored. An event with no data isn't
 *       dispatched. Every event type is passed on: a provider dispatches on SseEvent::type and
 *       ignores the types it doesn't know, which APIs add without notice (`ping`, for one).
 */
class SseParser {
public:
    using Handler = std::function<void(const SseEvent&)>;

    /**
     * @param on_event called for each complete event.
     * @param max_event_bytes the largest event (or line) to accept from the server.
     */
    explicit SseParser(Handler on_event, std::size_t max_event_bytes = std::size_t{16} << 20);

    /** @brief Parses @p chunk, calling the handler for every event it completes.
     *  @throws std::length_error if a line or an event passes max_event_bytes. */
    void feed(std::string_view chunk);

    /** @brief The end of the stream. An event without its closing blank line is discarded, as the
     *         specification requires. */
    void finish();

    /** @brief The reconnection time from the last valid `retry:` field, if any. */
    [[nodiscard]] std::optional<std::chrono::milliseconds> retry() const { return retry_; }

private:
    void process_line(std::string_view line);
    void dispatch();

    Handler on_event_;
    std::size_t max_event_bytes_;
    std::string line_;
    std::string data_;
    std::string type_;
    std::string last_event_id_;
    std::optional<std::chrono::milliseconds> retry_;
    bool after_cr_ = false;  // the last chunk ended in CR, so a leading LF finishes that line
    bool started_ = false;   // past the position where a byte-order mark may appear
    std::size_t bom_matched_ = 0;
    bool has_data_ = false;
};

}  // namespace pulsatrix::orch
