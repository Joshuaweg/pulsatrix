// AGT-3: the Anthropic and OpenAI-compatible providers and the tool loop, against a scripted
// transport that replays streams shaped like the APIs' documented examples and records each
// request. No network. One test calls the real Anthropic API, only when PULSATRIX_ORCH_LIVE=1.
#include "pulsatrix/orch/provider.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/orch/tool_loop.hpp"

namespace pulsatrix::orch {
namespace {

using namespace std::chrono_literals;

// ---- a scripted transport -------------------------------------------------------------------

struct Scripted {
    int status = 200;
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::string> chunks;
    std::optional<HttpError::Kind> fail_after_chunks;  // the connection breaks after the chunks
};

class FakeTransport : public HttpTransport {
public:
    std::deque<Scripted> script;
    std::vector<HttpRequest> requests;

    HttpResponse send(const HttpRequest& request) override {
        return stream(request, [](std::string_view) { return true; });
    }

    HttpResponse stream(const HttpRequest& request, const HttpChunkHandler& on_chunk) override {
        requests.push_back(request);
        if (script.empty()) {
            throw std::logic_error("FakeTransport: no scripted response left");
        }
        Scripted s = std::move(script.front());
        script.pop_front();
        HttpResponse response;
        response.status = s.status;
        response.headers = s.headers;
        for (const std::string& chunk : s.chunks) {
            if (response.ok()) {
                if (!on_chunk(chunk)) {
                    return response;
                }
            } else {
                response.body += chunk;
            }
        }
        if (s.fail_after_chunks) {
            throw HttpError(*s.fail_after_chunks, "scripted failure");
        }
        return response;
    }

    std::string name() const override { return "fake"; }

    JsonValue body(std::size_t i) const { return ParseJson(requests.at(i).body); }
};

std::string Sse(const std::string& type, const std::string& json) { return "event: " + type + "\ndata: " + json + "\n\n"; }

// An answer that thinks, says something, calls a tool, and passes new event and block types.
std::vector<std::string> AnthropicToolUseStream() {
    return {
        Sse("message_start",
            R"({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":25,"cache_creation_input_tokens":600,"cache_read_input_tokens":0,"output_tokens":1}}})"),
        Sse("content_block_start", R"({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"The user wants "}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"a sum."}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM"}})"),
        Sse("content_block_stop", R"({"type":"content_block_stop","index":0})"),
        "event: ping\ndata: {\"type\": \"ping\"}\n\n",
        Sse("brand_new_event", R"({"type":"brand_new_event","whatever":1})"),
        Sse("content_block_start", R"({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me "}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"add them."}})"),
        Sse("content_block_stop", R"({"type":"content_block_stop","index":1})"),
        Sse("content_block_start", R"({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_01","name":"add","input":{}}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":""}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\": 2,"}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":" \"b\": 3}"}})"),
        Sse("content_block_stop", R"({"type":"content_block_stop","index":2})"),
        Sse("message_delta", R"({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}})"),
        Sse("message_stop", R"({"type":"message_stop"})"),
    };
}

std::vector<std::string> AnthropicTextStream(const std::string& text) {
    return {
        Sse("message_start", R"({"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":10,"cache_read_input_tokens":600,"output_tokens":1}}})"),
        Sse("content_block_start", R"({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":")" + text + R"("}})"),
        Sse("content_block_stop", R"({"type":"content_block_stop","index":0})"),
        Sse("message_delta", R"({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}})"),
        Sse("message_stop", R"({"type":"message_stop"})"),
    };
}

Scripted Ok(std::vector<std::string> chunks) { return {200, {{"request-id", "req_ok"}}, std::move(chunks), std::nullopt}; }

Scripted Error(int status, const std::string& body, std::vector<std::pair<std::string, std::string>> headers = {}) {
    return {status, std::move(headers), {body}, std::nullopt};
}

ToolSpec AddTool() {
    ToolSpec t;
    t.name = "add";
    t.description = "Adds two integers a and b and returns their sum.";
    t.input_schema = ParseJson(R"({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                  "required": ["a", "b"], "additionalProperties": false})");
    t.strict = true;
    return t;
}

ToolSpec EchoTool() {
    ToolSpec t;
    t.name = "echo";
    t.description = "Returns its input.";
    t.input_schema = ParseJson(R"({"type": "object", "properties": {"text": {"type": "string"}}})");
    return t;
}

ChatRequest AddRequest() {
    ChatRequest r;
    r.model = "claude-opus-5-5";
    r.system = "You are a careful calculator.";
    r.tools = {EchoTool(), AddTool()};
    r.messages.push_back(Message::User("What is 2 + 3?"));
    return r;
}

struct Harness {
    std::shared_ptr<FakeTransport> transport = std::make_shared<FakeTransport>();
    std::vector<std::chrono::milliseconds> sleeps;

    RetryPolicy retry() {
        RetryPolicy p;
        p.sleep = [this](std::chrono::milliseconds d) { sleeps.push_back(d); };
        return p;
    }
    std::unique_ptr<Provider> anthropic() {
        AnthropicOptions o;
        o.api_key = Secret("sk-ant-test-key");
        o.retry = retry();
        return MakeAnthropicProvider(std::move(o), transport);
    }
    std::unique_ptr<Provider> openai(bool with_key = false) {
        OpenAiCompatibleOptions o;
        o.retry = retry();
        if (with_key) {
            o.api_key = Secret("sk-local");
        }
        return MakeOpenAiCompatibleProvider(std::move(o), transport);
    }
};

ToolOutcome Add(const ToolCall& call) {
    return {std::to_string(call.input.find("a")->as_int64() + call.input.find("b")->as_int64())};
}

// ---- Anthropic: the request ----------------------------------------------------------------

TEST(OrchAnthropicTest, BuildsTheRequestTheApiExpects) {
    Harness h;
    h.transport->script.push_back(Ok(AnthropicTextStream("5")));
    (void)h.anthropic()->chat(AddRequest());

    const HttpRequest& http = h.transport->requests.at(0);
    EXPECT_EQ(http.method, "POST");
    EXPECT_EQ(http.url, "https://api.anthropic.com/v1/messages");
    const std::string described = http.describe();
    EXPECT_NE(described.find("x-api-key: <redacted>"), std::string::npos);
    EXPECT_EQ(described.find("sk-ant-test-key"), std::string::npos);
    EXPECT_NE(described.find("anthropic-version: 2023-06-01"), std::string::npos);

    const JsonValue body = h.transport->body(0);
    EXPECT_EQ(body.find("model")->as_string(), "claude-opus-5-5");
    EXPECT_TRUE(body.find("stream")->as_bool());
    EXPECT_EQ(body.find("max_tokens")->as_int64(), 4096);
    // Caching: automatic for the conversation, explicit on the system prompt and the last tool.
    EXPECT_EQ(body.find("cache_control")->find("type")->as_string(), "ephemeral");
    const JsonValue& system = body.find("system")->as_array().at(0);
    EXPECT_EQ(system.find("text")->as_string(), "You are a careful calculator.");
    EXPECT_NE(system.find("cache_control"), nullptr);
    const JsonValue::Array& tools = body.find("tools")->as_array();
    ASSERT_EQ(tools.size(), 2u);
    EXPECT_EQ(tools[0].find("cache_control"), nullptr);
    EXPECT_NE(tools[1].find("cache_control"), nullptr);
    EXPECT_TRUE(tools[1].find("strict")->as_bool());
    EXPECT_EQ(tools[0].find("strict"), nullptr);
    // Never a forced tool choice: current models answer that with HTTP 400.
    EXPECT_EQ(body.find("tool_choice"), nullptr);
    const JsonValue& message = body.find("messages")->as_array().at(0);
    EXPECT_EQ(message.find("role")->as_string(), "user");
    EXPECT_EQ(message.find("content")->as_array().at(0).find("text")->as_string(), "What is 2 + 3?");
}

TEST(OrchAnthropicTest, ToolCallsCanBeTurnedOffAndCachingToo) {
    Harness h;
    h.transport->script.push_back(Ok(AnthropicTextStream("5")));
    AnthropicOptions o;
    o.api_key = Secret("k");
    o.prompt_caching = false;
    o.betas = {"feature-a-2026-01-01", "feature-b-2026-02-01"};
    ChatRequest request = AddRequest();
    request.allow_tool_calls = false;
    (void)MakeAnthropicProvider(std::move(o), h.transport)->chat(request);
    const JsonValue body = h.transport->body(0);
    EXPECT_EQ(body.find("tool_choice")->find("type")->as_string(), "none");
    EXPECT_EQ(body.find("cache_control"), nullptr);
    EXPECT_EQ(body.find("system")->as_array().at(0).find("cache_control"), nullptr);
    EXPECT_NE(h.transport->requests[0].describe().find("anthropic-beta: feature-a-2026-01-01,feature-b-2026-02-01"),
              std::string::npos);
}

TEST(OrchAnthropicTest, RefusesAnInvalidRequestBeforeSendingIt) {
    Harness h;
    ChatRequest r = AddRequest();
    r.tools[0].name = "has space";
    EXPECT_THROW((void)h.anthropic()->chat(r), std::invalid_argument);
    r = AddRequest();
    r.model.clear();
    EXPECT_THROW((void)h.anthropic()->chat(r), std::invalid_argument);
    EXPECT_TRUE(h.transport->requests.empty());
    AnthropicOptions no_key;
    EXPECT_THROW((void)MakeAnthropicProvider(no_key, h.transport), std::invalid_argument);
}

// ---- Anthropic: the stream -----------------------------------------------------------------

TEST(OrchAnthropicTest, AssemblesAStreamedAnswer) {
    Harness h;
    h.transport->script.push_back(Ok(AnthropicToolUseStream()));
    std::string text;
    std::string thinking;
    StreamCallbacks callbacks{[&](std::string_view t) { text += t; }, [&](std::string_view t) { thinking += t; }};
    ChatResponse r = h.anthropic()->chat(AddRequest(), callbacks);

    EXPECT_EQ(text, "Let me add them.");
    EXPECT_EQ(thinking, "The user wants a sum.");
    EXPECT_EQ(r.stop_reason, StopReason::ToolUse);
    EXPECT_EQ(r.stop_reason_text, "tool_use");
    EXPECT_EQ(r.model, "claude-opus-5-5");
    EXPECT_EQ(r.request_id, "req_ok");
    EXPECT_EQ(r.usage.input_tokens, 25);
    EXPECT_EQ(r.usage.output_tokens, 89);  // message_delta's cumulative count wins
    EXPECT_EQ(r.usage.cache_creation_input_tokens, 600);

    ASSERT_EQ(r.message.content.size(), 3u);
    EXPECT_EQ(r.message.content[0].type, ContentBlock::Type::Thinking);
    EXPECT_EQ(WriteJsonCompact(r.message.content[0].raw),
              R"({"type":"thinking","thinking":"The user wants a sum.","signature":"EqQBCgIYAhIM"})");
    EXPECT_EQ(r.message.text(), "Let me add them.");
    ASSERT_EQ(r.message.tool_uses().size(), 1u);
    const ContentBlock& call = *r.message.tool_uses()[0];
    EXPECT_EQ(call.id, "toolu_01");
    EXPECT_EQ(call.name, "add");
    EXPECT_EQ(WriteJsonCompact(call.input), R"({"a":2,"b":3})");
}

TEST(OrchAnthropicTest, KeepsBlocksItDoesNotKnowAndSendsThemBack) {
    Harness h;
    h.transport->script.push_back(Ok({
        Sse("message_start", R"({"type":"message_start","message":{"model":"m","usage":{"input_tokens":1}}})"),
        Sse("content_block_start", R"({"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"EmwKAhgB"}})"),
        Sse("content_block_stop", R"({"type":"content_block_stop","index":0})"),
        Sse("content_block_start", R"({"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}})"),
        Sse("content_block_delta", R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\": \"pulsatrix\"}"}})"),
        Sse("content_block_stop", R"({"type":"content_block_stop","index":1})"),
        Sse("message_delta", R"({"type":"message_delta","delta":{"stop_reason":"pause_turn"}})"),
        Sse("message_stop", R"({"type":"message_stop"})"),
    }));
    ChatResponse r = h.anthropic()->chat(AddRequest());
    EXPECT_EQ(r.stop_reason, StopReason::PauseTurn);
    ASSERT_EQ(r.message.content.size(), 2u);
    EXPECT_EQ(r.message.content[0].type, ContentBlock::Type::Opaque);
    EXPECT_EQ(WriteJsonCompact(r.message.content[0].raw), R"({"type":"redacted_thinking","data":"EmwKAhgB"})");
    EXPECT_EQ(WriteJsonCompact(r.message.content[1].raw),
              R"({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"pulsatrix"}})");

    // Sent back as received; another provider's thinking isn't sent.
    ChatRequest next = AddRequest();
    next.messages.push_back(r.message);
    ContentBlock foreign;
    foreign.type = ContentBlock::Type::Thinking;
    foreign.text = "local reasoning";
    foreign.origin = "openai-compatible";
    next.messages.back().content.push_back(foreign);
    h.transport->script.push_back(Ok(AnthropicTextStream("ok")));
    (void)h.anthropic()->chat(next);
    const JsonValue second_request = h.transport->body(1);
    const JsonValue::Array& sent = second_request.find("messages")->as_array().at(1).find("content")->as_array();
    ASSERT_EQ(sent.size(), 2u);
    EXPECT_EQ(sent[0].find("type")->as_string(), "redacted_thinking");
    EXPECT_EQ(sent[1].find("type")->as_string(), "server_tool_use");
}

TEST(OrchAnthropicTest, AStreamThatStopsEarlyIsAnError) {
    Harness h;
    std::vector<std::string> cut = AnthropicTextStream("x");
    cut.pop_back();  // no message_stop
    for (int i = 0; i < 3; ++i) {
        h.transport->script.push_back(Ok(cut));
    }
    EXPECT_THROW((void)h.anthropic()->chat(AddRequest()), ProviderError);
    EXPECT_EQ(h.transport->requests.size(), 3u);  // retried, as a broken connection is
}

// ---- Anthropic: errors and retries ---------------------------------------------------------

TEST(OrchAnthropicRetryTest, RetriesAnOverloadWithJitteredBackoff) {
    Harness h;
    const std::string overloaded = R"({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_x"})";
    h.transport->script.push_back(Error(529, overloaded));
    h.transport->script.push_back(Error(500, R"({"type":"error","error":{"type":"api_error","message":"boom"}})"));
    h.transport->script.push_back(Ok(AnthropicTextStream("5")));
    ChatResponse r = h.anthropic()->chat(AddRequest());
    EXPECT_EQ(r.message.text(), "5");
    EXPECT_EQ(h.transport->requests.size(), 3u);
    ASSERT_EQ(h.sleeps.size(), 2u);
    EXPECT_LE(h.sleeps[0], 1000ms);  // a random fraction of 1 s, then of 2 s
    EXPECT_LE(h.sleeps[1], 2000ms);
}

TEST(OrchAnthropicRetryTest, GivesUpAfterMaxRetries) {
    Harness h;
    for (int i = 0; i < 3; ++i) {
        h.transport->script.push_back(Error(529, R"({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_x"})"));
    }
    try {
        (void)h.anthropic()->chat(AddRequest());
        ADD_FAILURE() << "didn't throw";
    } catch (const ProviderError& e) {
        EXPECT_EQ(e.status(), 529);
        EXPECT_EQ(e.type(), "overloaded_error");
        EXPECT_EQ(e.request_id(), "req_x");
        EXPECT_NE(std::string(e.what()).find("req_x"), std::string::npos);
    }
    EXPECT_EQ(h.transport->requests.size(), 3u);
}

TEST(OrchAnthropicRetryTest, HonorsRetryAfter) {
    Harness h;
    h.transport->script.push_back(Error(429, R"({"type":"error","error":{"type":"rate_limit_error","message":"slow down"}})",
                                        {{"retry-after", "3"}}));
    h.transport->script.push_back(Ok(AnthropicTextStream("5")));
    (void)h.anthropic()->chat(AddRequest());
    ASSERT_EQ(h.sleeps.size(), 1u);
    EXPECT_EQ(h.sleeps[0], 3000ms);

    Harness too_long;
    too_long.transport->script.push_back(Error(429, R"({"type":"error","error":{"type":"rate_limit_error","message":"x"}})",
                                               {{"retry-after", "3600"}}));
    EXPECT_THROW((void)too_long.anthropic()->chat(AddRequest()), ProviderError);
    EXPECT_TRUE(too_long.sleeps.empty());
}

TEST(OrchAnthropicRetryTest, NeverRetriesTheSpendCap) {
    Harness h;
    h.transport->script.push_back(Error(
        429,
        R"({"type":"error","error":{"type":"rate_limit_error","message":"You have reached your API usage limits.","details":{"error_code":"enforced_spend_limit_reached"}},"request_id":"req_cap"})"));
    try {
        (void)h.anthropic()->chat(AddRequest());
        ADD_FAILURE() << "didn't throw";
    } catch (const ProviderError& e) {
        EXPECT_EQ(e.status(), 429);
        EXPECT_NE(std::string(e.what()).find("enforced_spend_limit_reached"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("not retried"), std::string::npos);
    }
    EXPECT_EQ(h.transport->requests.size(), 1u);
    EXPECT_TRUE(h.sleeps.empty());
}

TEST(OrchAnthropicRetryTest, NeverRetriesAClientError) {
    Harness h;
    h.transport->script.push_back(Error(
        400, R"({"type":"error","error":{"type":"invalid_request_error","message":"tool_choice: type \"tool\" and \"any\" are not supported for this model."}})",
        {{"request-id", "req_hdr"}}));
    try {
        (void)h.anthropic()->chat(AddRequest());
        ADD_FAILURE() << "didn't throw";
    } catch (const ProviderError& e) {
        EXPECT_EQ(e.status(), 400);
        EXPECT_EQ(e.type(), "invalid_request_error");
        EXPECT_EQ(e.request_id(), "req_hdr");  // from the header when the body has none
    }
    EXPECT_EQ(h.transport->requests.size(), 1u);
}

TEST(OrchAnthropicRetryTest, RetriesAnOverloadInTheStreamOnlyBeforeAnythingWasShown) {
    std::vector<std::string> overloaded_first = {
        Sse("message_start", R"({"type":"message_start","message":{"model":"m","usage":{}}})"),
        Sse("error", R"({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}})"),
    };
    Harness h;
    h.transport->script.push_back(Ok(overloaded_first));
    h.transport->script.push_back(Ok(AnthropicTextStream("5")));
    EXPECT_EQ(h.anthropic()->chat(AddRequest()).message.text(), "5");
    EXPECT_EQ(h.transport->requests.size(), 2u);

    // Once text has reached the caller, a retry would repeat it: the error goes to the caller.
    std::vector<std::string> overloaded_late = AnthropicTextStream("partial");
    overloaded_late.resize(3);
    overloaded_late.push_back(Sse("error", R"({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}})"));
    Harness late;
    late.transport->script.push_back(Ok(overloaded_late));
    std::string shown;
    try {
        (void)late.anthropic()->chat(AddRequest(), {[&](std::string_view t) { shown += t; }, {}});
        ADD_FAILURE() << "didn't throw";
    } catch (const ProviderError& e) {
        EXPECT_EQ(e.type(), "overloaded_error");
        EXPECT_NE(std::string(e.what()).find("already streamed"), std::string::npos);
    }
    EXPECT_EQ(shown, "partial");
    EXPECT_EQ(late.transport->requests.size(), 1u);
}

TEST(OrchAnthropicRetryTest, RetriesABrokenConnection) {
    Harness h;
    h.transport->script.push_back({200, {}, {}, HttpError::Kind::Connect});
    h.transport->script.push_back({200, {}, {AnthropicTextStream("5")[0]}, HttpError::Kind::Protocol});
    h.transport->script.push_back(Ok(AnthropicTextStream("5")));
    EXPECT_EQ(h.anthropic()->chat(AddRequest()).message.text(), "5");
    EXPECT_EQ(h.transport->requests.size(), 3u);
}

TEST(OrchAnthropicRetryTest, ANonJsonErrorPageStillGivesAReadableError) {
    Harness h;
    h.transport->script.push_back(Error(403, "<html>Forbidden by proxy</html>"));
    try {
        (void)h.anthropic()->chat(AddRequest());
        ADD_FAILURE() << "didn't throw";
    } catch (const ProviderError& e) {
        EXPECT_NE(std::string(e.what()).find("Forbidden by proxy"), std::string::npos);
    }
}

TEST(OrchAnthropicTest, FromEnvNeedsTheKey) {
#ifdef _WIN32
    _putenv_s("ANTHROPIC_API_KEY", "");
#else
    const char* saved = std::getenv("ANTHROPIC_API_KEY");
    const std::string restore = saved != nullptr ? saved : "";
    ::unsetenv("ANTHROPIC_API_KEY");
#endif
    EXPECT_THROW((void)AnthropicOptions::FromEnv(), std::runtime_error);
#ifndef _WIN32
    if (!restore.empty()) {
        ::setenv("ANTHROPIC_API_KEY", restore.c_str(), 1);
    }
#endif
}

// ---- the tool loop -------------------------------------------------------------------------

TEST(OrchToolLoopTest, RunsToolsAndSendsResultsBack) {
    Harness h;
    h.transport->script.push_back(Ok(AnthropicToolUseStream()));
    h.transport->script.push_back(Ok(AnthropicTextStream("2 + 3 = 5.")));
    std::vector<Message> added;
    ToolLoopOptions options;
    options.on_message = [&](const Message& m) { added.push_back(m); };
    ToolLoopResult result = RunToolLoop(*h.anthropic(), AddRequest(), Add, options);

    EXPECT_EQ(result.turns, 2);
    EXPECT_FALSE(result.hit_turn_limit);
    EXPECT_EQ(result.last.message.text(), "2 + 3 = 5.");
    EXPECT_EQ(result.messages.size(), 4u);  // question, call, result, answer
    EXPECT_EQ(added.size(), 3u);
    EXPECT_EQ(result.usage.input_tokens, 35);
    EXPECT_EQ(result.usage.cache_read_input_tokens, 600);

    // The second request: the assistant turn sent back with its thinking unchanged, then the
    // tool result first in the user message.
    const JsonValue second_request = h.transport->body(1);
    const JsonValue::Array& messages = second_request.find("messages")->as_array();
    ASSERT_EQ(messages.size(), 3u);
    const JsonValue::Array& assistant = messages[1].find("content")->as_array();
    EXPECT_EQ(WriteJsonCompact(assistant[0]), R"({"type":"thinking","thinking":"The user wants a sum.","signature":"EqQBCgIYAhIM"})");
    EXPECT_EQ(WriteJsonCompact(assistant[2]), R"({"type":"tool_use","id":"toolu_01","name":"add","input":{"a":2,"b":3}})");
    const JsonValue& tool_result = messages[2].find("content")->as_array().at(0);
    EXPECT_EQ(WriteJsonCompact(tool_result), R"({"type":"tool_result","tool_use_id":"toolu_01","content":"5"})");
}

TEST(OrchToolLoopTest, AToolThatThrowsIsReportedToTheModel) {
    Harness h;
    h.transport->script.push_back(Ok(AnthropicToolUseStream()));
    h.transport->script.push_back(Ok(AnthropicTextStream("Sorry.")));
    ToolLoopResult result = RunToolLoop(*h.anthropic(), AddRequest(), [](const ToolCall&) -> ToolOutcome {
        throw std::runtime_error("the calculator is broken");
    });
    const ContentBlock& sent = result.messages.at(2).content.at(0);
    EXPECT_EQ(sent.type, ContentBlock::Type::ToolResult);
    EXPECT_TRUE(sent.is_error);
    EXPECT_EQ(sent.text, "The tool failed: the calculator is broken");
    EXPECT_TRUE(h.transport->body(1).find("messages")->as_array().at(2).find("content")->as_array().at(0).find("is_error")->as_bool());
}

TEST(OrchToolLoopTest, StopsAtTheTurnLimit) {
    Harness h;
    for (int i = 0; i < 3; ++i) {
        h.transport->script.push_back(Ok(AnthropicToolUseStream()));
    }
    ToolLoopOptions options;
    options.max_turns = 3;
    ToolLoopResult result = RunToolLoop(*h.anthropic(), AddRequest(), Add, options);
    EXPECT_EQ(result.turns, 3);
    EXPECT_TRUE(result.hit_turn_limit);
    EXPECT_EQ(h.transport->requests.size(), 3u);
}

TEST(OrchToolLoopTest, ContinuesAPausedTurn) {
    Harness h;
    std::vector<std::string> paused = AnthropicTextStream("Searching...");
    paused[4] = Sse("message_delta", R"({"type":"message_delta","delta":{"stop_reason":"pause_turn"}})");
    h.transport->script.push_back(Ok(paused));
    h.transport->script.push_back(Ok(AnthropicTextStream("Done.")));
    ToolLoopResult result = RunToolLoop(*h.anthropic(), AddRequest(), Add);
    EXPECT_EQ(result.turns, 2);
    EXPECT_EQ(result.last.message.text(), "Done.");
    const JsonValue second_request = h.transport->body(1);
    const JsonValue::Array& messages = second_request.find("messages")->as_array();
    EXPECT_EQ(messages.back().find("role")->as_string(), "assistant");  // sent back as it was
}

// ---- OpenAI-compatible ---------------------------------------------------------------------

std::string Data(const std::string& json) { return "data: " + json + "\n\n"; }

TEST(OrchOpenAiTest, BuildsAChatCompletionsRequest) {
    Harness h;
    h.transport->script.push_back(Ok({Data(R"({"choices":[{"delta":{"content":"5"},"finish_reason":"stop"}]})"), "data: [DONE]\n\n"}));
    ChatRequest request = AddRequest();
    // A history with a tool call, its result, and an Anthropic thinking block that must not be sent.
    Message call;
    call.role = Message::Role::Assistant;
    ContentBlock thinking;
    thinking.type = ContentBlock::Type::Thinking;
    thinking.origin = "anthropic";
    thinking.raw = ParseJson(R"({"type":"thinking","thinking":"x","signature":"y"})");
    call.content = {thinking, ContentBlock::Text("Adding."), ContentBlock::ToolUse("call_9", "add", ParseJson(R"({"a":2,"b":3})"))};
    Message result;
    result.content = {ContentBlock::ToolResult("call_9", "5"), ContentBlock::Text("Thanks.")};
    request.messages.push_back(call);
    request.messages.push_back(result);
    (void)h.openai(/*with_key=*/true)->chat(request);

    const HttpRequest& http = h.transport->requests.at(0);
    EXPECT_EQ(http.url, "http://127.0.0.1:8080/v1/chat/completions");
    EXPECT_NE(http.describe().find("authorization: <redacted>"), std::string::npos);
    EXPECT_EQ(http.describe().find("sk-local"), std::string::npos);
    EXPECT_EQ(http.headers[0].value, "Bearer sk-local");

    const JsonValue body = h.transport->body(0);
    EXPECT_TRUE(body.find("stream_options")->find("include_usage")->as_bool());
    EXPECT_EQ(body.find("tool_choice")->as_string(), "auto");
    const JsonValue& tool = body.find("tools")->as_array().at(1);
    EXPECT_EQ(tool.find("type")->as_string(), "function");
    EXPECT_EQ(tool.find("function")->find("name")->as_string(), "add");
    EXPECT_TRUE(tool.find("function")->find("strict")->as_bool());
    const JsonValue::Array& m = body.find("messages")->as_array();
    ASSERT_EQ(m.size(), 5u);  // system, user, assistant, tool, user
    EXPECT_EQ(m[0].find("role")->as_string(), "system");
    EXPECT_EQ(m[2].find("content")->as_string(), "Adding.");
    const JsonValue& c = m[2].find("tool_calls")->as_array().at(0);
    EXPECT_EQ(c.find("id")->as_string(), "call_9");
    EXPECT_EQ(c.find("function")->find("arguments")->as_string(), R"({"a":2,"b":3})");
    EXPECT_EQ(m[3].find("role")->as_string(), "tool");
    EXPECT_EQ(m[3].find("tool_call_id")->as_string(), "call_9");
    EXPECT_EQ(m[4].find("content")->as_string(), "Thanks.");
}

TEST(OrchOpenAiTest, AssemblesStreamedToolCalls) {
    Harness h;
    // Two calls whose fragments interleave, a reasoning delta, then usage after finish_reason.
    h.transport->script.push_back(Ok({
        Data(R"({"model":"qwen2.5-0.5b","choices":[{"delta":{"role":"assistant","reasoning_content":"Two sums."}}]})"),
        Data(R"({"choices":[{"delta":{"content":"Adding."}}]})"),
        Data(R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"add","arguments":""}}]}}]})"),
        Data(R"({"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"add","arguments":"{\"a\":1,"}}]}}]})"),
        Data(R"({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":2,\"b\":3}"}}]}}]})"),
        Data(R"({"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"b\":1}"}}]}}]})"),
        Data(R"({"choices":[{"delta":{},"finish_reason":"tool_calls"}]})"),
        Data(R"({"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":12,"prompt_tokens_details":{"cached_tokens":32}}})"),
        "data: [DONE]\n\n",
    }));
    std::string thinking;
    ChatResponse r = h.openai()->chat(AddRequest(), {{}, [&](std::string_view t) { thinking += t; }});
    EXPECT_EQ(thinking, "Two sums.");
    EXPECT_EQ(r.stop_reason, StopReason::ToolUse);
    EXPECT_EQ(r.model, "qwen2.5-0.5b");
    EXPECT_EQ(r.usage.input_tokens, 40);
    EXPECT_EQ(r.usage.output_tokens, 12);
    EXPECT_EQ(r.usage.cache_read_input_tokens, 32);
    EXPECT_EQ(r.message.text(), "Adding.");
    std::vector<const ContentBlock*> calls = r.message.tool_uses();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0]->id, "call_a");
    EXPECT_EQ(WriteJsonCompact(calls[0]->input), R"({"a":2,"b":3})");
    EXPECT_EQ(calls[1]->id, "call_1");  // the server sent no id
    EXPECT_EQ(WriteJsonCompact(calls[1]->input), R"({"a":1,"b":1})");
}

TEST(OrchOpenAiTest, KeepsArgumentsThatArentJsonForTheToolToReport) {
    Harness h;
    h.transport->script.push_back(Ok({
        Data(R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"add","arguments":"{a: 2"}}]},"finish_reason":"stop"}]})"),
        "data: [DONE]\n\n",
    }));
    ChatResponse r = h.openai()->chat(AddRequest());
    EXPECT_EQ(r.stop_reason, StopReason::ToolUse);  // "stop" with calls still means calls
    ASSERT_EQ(r.message.tool_uses().size(), 1u);
    EXPECT_EQ(r.message.tool_uses()[0]->input.type(), JsonValue::Type::String);
    EXPECT_EQ(r.message.tool_uses()[0]->input.as_string(), "{a: 2");
}

TEST(OrchOpenAiTest, RetriesA429WithoutRetryAfterAndReportsStreamErrors) {
    Harness h;
    h.transport->script.push_back(Error(429, R"({"error":{"message":"busy","type":"rate_limit"}})"));
    h.transport->script.push_back(Ok({Data(R"({"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]})")}));
    EXPECT_EQ(h.openai()->chat(AddRequest()).message.text(), "ok");
    EXPECT_EQ(h.transport->requests.size(), 2u);

    Harness e;
    e.transport->script.push_back(Ok({Data(R"({"error":{"message":"context too long"}})")}));
    try {
        (void)e.openai()->chat(AddRequest());
        ADD_FAILURE() << "didn't throw";
    } catch (const ProviderError& error) {
        EXPECT_NE(std::string(error.what()).find("context too long"), std::string::npos);
    }
}

TEST(OrchOpenAiTest, RunsTheToolLoop) {
    Harness h;
    h.transport->script.push_back(Ok({
        Data(R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}}]}}]})"),
        Data(R"({"choices":[{"delta":{},"finish_reason":"tool_calls"}]})"),
    }));
    h.transport->script.push_back(Ok({Data(R"({"choices":[{"delta":{"content":"5"},"finish_reason":"stop"}]})")}));
    ToolLoopResult result = RunToolLoop(*h.openai(), AddRequest(), Add);
    EXPECT_EQ(result.last.message.text(), "5");
    const JsonValue second_request = h.transport->body(1);
    const JsonValue::Array& m = second_request.find("messages")->as_array();
    EXPECT_EQ(m.back().find("role")->as_string(), "tool");
    EXPECT_EQ(m.back().find("content")->as_string(), "5");
}

// ---- the real API, only when asked ---------------------------------------------------------

TEST(OrchLiveTest, AnthropicToolLoop) {
    const char* live = std::getenv("PULSATRIX_ORCH_LIVE");
    if (live == nullptr || std::string(live) != "1" || !Secret::FromEnv("ANTHROPIC_API_KEY") || !HttpsSupported(HttpBackend::Httplib)) {
        GTEST_SKIP() << "set PULSATRIX_ORCH_LIVE=1 and ANTHROPIC_API_KEY to call the real API (a few cents)";
    }
    std::unique_ptr<Provider> claude = MakeAnthropicProvider(AnthropicOptions::FromEnv());
    ChatRequest request = AddRequest();
    request.model = "claude-haiku-5-5";
    request.max_tokens = 1024;
    request.messages = {Message::User("Use the add tool to add 1234 and 4321, then reply with just the sum.")};
    ToolLoopResult result = RunToolLoop(*claude, request, Add);
    EXPECT_GE(result.turns, 2);
    EXPECT_NE(result.last.message.text().find("5555"), std::string::npos) << result.last.message.text();
}

}  // namespace
}  // namespace pulsatrix::orch
