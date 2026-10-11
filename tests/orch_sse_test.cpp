// AGT-2: the server-sent-events parser, against the WHATWG HTML specification's rules.
#include "pulsatrix/orch/sse.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix::orch {
namespace {

std::vector<SseEvent> ParseInChunks(const std::string& text, std::size_t chunk) {
    std::vector<SseEvent> events;
    SseParser parser([&](const SseEvent& e) { events.push_back(e); });
    for (std::size_t i = 0; i < text.size(); i += chunk) {
        parser.feed(std::string_view(text).substr(i, chunk));
    }
    parser.finish();
    return events;
}

std::vector<SseEvent> Parse(const std::string& text) { return ParseInChunks(text, text.size() + 1); }

// An Anthropic-style stream, with a comment, a ping and a multi-line data field.
const std::string kStream =
    "\xEF\xBB\xBF"
    ": keep-alive\r\n"
    "event: message_start\r\n"
    "data: {\"type\": \"message_start\"}\r\n"
    "\r\n"
    "event: ping\n"
    "data: {}\n"
    "\n"
    "id: 7\r"
    "data: first line\r"
    "data:second line\r"
    "\r"
    "event: message_stop\n"
    "data: {\"type\": \"message_stop\"}\n"
    "\n";

void ExpectStream(const std::vector<SseEvent>& events) {
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(events[0].type, "message_start");
    EXPECT_EQ(events[0].data, "{\"type\": \"message_start\"}");
    EXPECT_EQ(events[1].type, "ping");
    EXPECT_EQ(events[1].data, "{}");
    EXPECT_EQ(events[2].type, "message");
    EXPECT_EQ(events[2].data, "first line\nsecond line");
    EXPECT_EQ(events[2].last_event_id, "7");
    EXPECT_EQ(events[3].type, "message_stop");
    EXPECT_EQ(events[3].last_event_id, "7");  // the last ID carries over
}

TEST(OrchSseTest, ParsesEveryLineEndingAndSkipsCommentsAndTheBom) { ExpectStream(Parse(kStream)); }

TEST(OrchSseTest, ChunkBoundariesDontMatter) {
    // Every chunk size splits the BOM, CRLF pairs and fields somewhere.
    for (std::size_t chunk = 1; chunk <= kStream.size(); ++chunk) {
        SCOPED_TRACE("chunk size " + std::to_string(chunk));
        ExpectStream(ParseInChunks(kStream, chunk));
    }
}

TEST(OrchSseTest, FieldRules) {
    // One space after the colon is removed, a second is kept; a line without a colon is a field
    // with an empty value; unknown fields are ignored.
    std::vector<SseEvent> events = Parse("data:  two spaces\nfoo: bar\ndata\n\n");
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].data, " two spaces\n");
}

TEST(OrchSseTest, AnEventWithoutDataIsNotDispatchedAndResetsTheType) {
    std::vector<SseEvent> events = Parse("event: lonely\n\ndata: x\n\n");
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].type, "message");
    EXPECT_EQ(events[0].data, "x");
}

TEST(OrchSseTest, AnUnfinishedEventAtTheEndIsDropped) {
    EXPECT_TRUE(Parse("data: never finished\n").empty());
    EXPECT_TRUE(Parse("data: no newline").empty());
}

TEST(OrchSseTest, IdAndRetry) {
    std::vector<SseEvent> events;
    SseParser parser([&](const SseEvent& e) { events.push_back(e); });
    parser.feed(std::string("id: a\0b\ndata: 1\n\n", 17));  // an id with a NUL is ignored
    parser.feed("retry: 1500\nretry: 12x\ndata: 2\n\n");
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].last_event_id, "");
    ASSERT_TRUE(parser.retry().has_value());
    EXPECT_EQ(parser.retry()->count(), 1500);  // 12x isn't digits, so the earlier value stays
}

TEST(OrchSseTest, ABomOnlyCountsAtTheStart) {
    std::vector<SseEvent> events = Parse("\xEF\xBB" "data: x\n\n");  // two of the three BOM bytes
    EXPECT_TRUE(events.empty());  // the line is "\xEF\xBBdata: x", an unknown field
    events = Parse("data: \xEF\xBB\xBF\n\n");
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].data, "\xEF\xBB\xBF");
}

TEST(OrchSseTest, RefusesOversizedLinesAndEvents) {
    SseParser lines([](const SseEvent&) {}, 16);
    EXPECT_THROW(lines.feed(std::string(17, 'x')), std::length_error);
    SseParser events([](const SseEvent&) {}, 16);
    EXPECT_NO_THROW(events.feed("data: 0123456\n"));
    EXPECT_THROW(events.feed("data: 0123456789\n"), std::length_error);
}

}  // namespace
}  // namespace pulsatrix::orch
