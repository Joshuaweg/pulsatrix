// The Anthropic Messages API (POST /v1/messages), always streamed. Follows the API rules in
// docs/roadmap/research-notes.md (AGT-3): no forced tool choice, tool results first, no retry of a
// spend-cap 429, backoff with jitter on 5xx, 529 and mid-stream overload, cached tools and system.
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "provider_internal.hpp"
#include "pulsatrix/orch/provider.hpp"
#include "pulsatrix/orch/sse.hpp"

namespace pulsatrix::orch {

AnthropicOptions AnthropicOptions::FromEnv() {
    std::optional<Secret> key = Secret::FromEnv("ANTHROPIC_API_KEY");
    if (!key) {
        throw std::runtime_error("ANTHROPIC_API_KEY isn't set");
    }
    AnthropicOptions options;
    options.api_key = std::move(*key);
    return options;
}

namespace {

using detail::EmptyArray;
using detail::EmptyObject;
using detail::IntOr;
using detail::ObjectMember;
using detail::StringOr;

constexpr const char* kOrigin = "anthropic";

JsonValue CacheControl() {
    JsonValue c = EmptyObject();
    c.add("type", "ephemeral");
    return c;
}

JsonValue BlockToJson(const ContentBlock& b) {
    JsonValue j = EmptyObject();
    switch (b.type) {
        case ContentBlock::Type::Text:
            j.add("type", "text");
            j.add("text", b.text);
            break;
        case ContentBlock::Type::ToolUse:
            j.add("type", "tool_use");
            j.add("id", b.id);
            j.add("name", b.name);
            j.add("input", b.input.type() == JsonValue::Type::Object ? b.input : EmptyObject());
            break;
        case ContentBlock::Type::ToolResult:
            j.add("type", "tool_result");
            j.add("tool_use_id", b.id);
            j.add("content", b.text);
            if (b.is_error) {
                j.add("is_error", true);
            }
            break;
        case ContentBlock::Type::Thinking:
        case ContentBlock::Type::Opaque:
            return b.raw;  // exactly as received
    }
    return j;
}

bool Sendable(const ContentBlock& b) {
    switch (b.type) {
        case ContentBlock::Type::Text:
            return !b.text.empty();  // the API rejects empty text blocks
        case ContentBlock::Type::Thinking:
        case ContentBlock::Type::Opaque:
            return b.origin == kOrigin && b.raw.type() == JsonValue::Type::Object;
        default:
            return true;
    }
}

StopReason ParseStopReason(const std::string& s) {
    if (s == "end_turn") return StopReason::EndTurn;
    if (s == "tool_use") return StopReason::ToolUse;
    if (s == "max_tokens") return StopReason::MaxTokens;
    if (s == "stop_sequence") return StopReason::StopSequence;
    if (s == "pause_turn") return StopReason::PauseTurn;
    if (s == "refusal") return StopReason::Refusal;
    return StopReason::Other;
}

void ReadUsage(const JsonValue& usage, Usage& out) {
    // message_delta's counts are cumulative, so the latest value of each field wins.
    out.input_tokens = IntOr(usage, "input_tokens", out.input_tokens);
    out.output_tokens = IntOr(usage, "output_tokens", out.output_tokens);
    out.cache_creation_input_tokens = IntOr(usage, "cache_creation_input_tokens", out.cache_creation_input_tokens);
    out.cache_read_input_tokens = IntOr(usage, "cache_read_input_tokens", out.cache_read_input_tokens);
}

// Builds the answer from the stream's events.
class StreamAssembler {
public:
    StreamAssembler(const StreamCallbacks& callbacks, bool& streamed) : callbacks_(callbacks), streamed_(streamed) {}

    void on_event(const SseEvent& event) {
        if (event.type == "ping") {
            return;
        }
        JsonValue data;
        try {
            data = ParseJson(event.data);
        } catch (const std::invalid_argument& e) {
            throw HttpError(HttpError::Kind::Protocol, std::string("anthropic: invalid JSON in a stream event: ") + e.what());
        }
        if (event.type == "message_start") {
            if (const JsonValue* message = ObjectMember(data, "message")) {
                response_.model = StringOr(*message, "model");
                if (const JsonValue* usage = ObjectMember(*message, "usage")) {
                    ReadUsage(*usage, response_.usage);
                }
            }
        } else if (event.type == "content_block_start") {
            start_block(data);
        } else if (event.type == "content_block_delta") {
            apply_delta(data);
        } else if (event.type == "content_block_stop") {
            finish_block(IntOr(data, "index", -1));
        } else if (event.type == "message_delta") {
            if (const JsonValue* delta = ObjectMember(data, "delta")) {
                response_.stop_reason_text = StringOr(*delta, "stop_reason", response_.stop_reason_text);
            }
            if (const JsonValue* usage = ObjectMember(data, "usage")) {
                ReadUsage(*usage, response_.usage);
            }
        } else if (event.type == "message_stop") {
            stopped_ = true;
        } else if (event.type == "error") {
            const JsonValue* error = ObjectMember(data, "error");
            const std::string type = error != nullptr ? StringOr(*error, "type") : "";
            const std::string message = error != nullptr ? StringOr(*error, "message") : event.data;
            ProviderError failure("anthropic: error in the stream: " + type + ": " + message, 0, type, "");
            if (type == "overloaded_error" || type == "api_error") {
                throw detail::RetryableFailure{failure, std::nullopt};
            }
            throw failure;
        }
        // Any other event type is new since this was written, and skipped.
    }

    ChatResponse finish(const std::string& request_id) {
        if (!stopped_) {
            throw HttpError(HttpError::Kind::Protocol, "anthropic: the stream ended before message_stop");
        }
        response_.message.role = Message::Role::Assistant;
        for (Partial& p : blocks_) {
            response_.message.content.push_back(std::move(p.block));
        }
        response_.stop_reason = ParseStopReason(response_.stop_reason_text);
        response_.request_id = request_id;
        return std::move(response_);
    }

private:
    struct Partial {
        ContentBlock block;
        std::string json;       // input_json_delta fragments
        std::string signature;  // a thinking block's signature_delta
        bool done = false;
    };

    Partial& at(int64_t index) {
        if (index < 0 || static_cast<std::size_t>(index) >= blocks_.size()) {
            throw HttpError(HttpError::Kind::Protocol, "anthropic: a stream event names unknown block " + std::to_string(index));
        }
        return blocks_[static_cast<std::size_t>(index)];
    }

    void start_block(const JsonValue& data) {
        const int64_t index = IntOr(data, "index", -1);
        if (index != static_cast<int64_t>(blocks_.size())) {
            throw HttpError(HttpError::Kind::Protocol, "anthropic: content blocks out of order");
        }
        const JsonValue* cb = ObjectMember(data, "content_block");
        if (cb == nullptr) {
            throw HttpError(HttpError::Kind::Protocol, "anthropic: content_block_start without a block");
        }
        Partial p;
        const std::string type = StringOr(*cb, "type");
        if (type == "text") {
            p.block = ContentBlock::Text(StringOr(*cb, "text"));
        } else if (type == "tool_use") {
            p.block = ContentBlock::ToolUse(StringOr(*cb, "id"), StringOr(*cb, "name"), EmptyObject());
        } else if (type == "thinking") {
            p.block.type = ContentBlock::Type::Thinking;
            p.block.text = StringOr(*cb, "thinking");
            p.signature = StringOr(*cb, "signature");
        } else {
            // redacted_thinking, server tool calls and results, and types added later: kept
            // whole and sent back as they came.
            p.block.type = ContentBlock::Type::Opaque;
            p.block.raw = *cb;
        }
        p.block.origin = kOrigin;
        blocks_.push_back(std::move(p));
    }

    void apply_delta(const JsonValue& data) {
        Partial& p = at(IntOr(data, "index", -1));
        const JsonValue* delta = ObjectMember(data, "delta");
        if (delta == nullptr) {
            return;
        }
        const std::string type = StringOr(*delta, "type");
        if (type == "text_delta") {
            const std::string text = StringOr(*delta, "text");
            p.block.text += text;
            if (callbacks_.on_text && !text.empty()) {
                streamed_ = true;
                callbacks_.on_text(text);
            }
        } else if (type == "input_json_delta") {
            p.json += StringOr(*delta, "partial_json");
        } else if (type == "thinking_delta") {
            const std::string thinking = StringOr(*delta, "thinking");
            p.block.text += thinking;
            if (callbacks_.on_thinking && !thinking.empty()) {
                streamed_ = true;
                callbacks_.on_thinking(thinking);
            }
        } else if (type == "signature_delta") {
            p.signature += StringOr(*delta, "signature");
        }
        // Other delta types (citations, for one) are skipped.
    }

    void finish_block(int64_t index) {
        Partial& p = at(index);
        p.done = true;
        if (p.block.type == ContentBlock::Type::ToolUse) {
            p.block.input = parse_input(p.json);
        } else if (p.block.type == ContentBlock::Type::Thinking) {
            JsonValue raw = EmptyObject();
            raw.add("type", "thinking");
            raw.add("thinking", p.block.text);
            raw.add("signature", p.signature);
            p.block.raw = std::move(raw);
        } else if (p.block.type == ContentBlock::Type::Opaque && !p.json.empty()) {
            p.block.raw = detail::WithMember(p.block.raw, "input", parse_input(p.json));  // a server tool call
        }
    }

    static JsonValue parse_input(const std::string& json) {
        if (json.empty()) {
            return EmptyObject();  // a tool without parameters streams no input
        }
        try {
            return ParseJson(json);
        } catch (const std::invalid_argument& e) {
            throw HttpError(HttpError::Kind::Protocol, std::string("anthropic: a tool call's input isn't valid JSON: ") + e.what());
        }
    }

    const StreamCallbacks& callbacks_;
    bool& streamed_;
    ChatResponse response_;
    std::vector<Partial> blocks_;
    bool stopped_ = false;
};

class AnthropicProvider final : public Provider {
public:
    AnthropicProvider(AnthropicOptions options, std::shared_ptr<HttpTransport> transport)
        : options_(std::move(options)), transport_(std::move(transport)), rng_(options_.retry.seed) {
        if (options_.api_key.empty()) {
            throw std::invalid_argument("AnthropicOptions: api_key is empty");
        }
        if (!transport_) {
            transport_ = MakeHttpTransport();
        }
    }

    ChatResponse chat(const ChatRequest& request, const StreamCallbacks& stream) override {
        detail::ValidateChatRequest(request);
        HttpRequest http;
        http.url = options_.base_url + "/v1/messages";
        http.read_timeout = options_.read_timeout;
        http.add_secret_header("x-api-key", options_.api_key);
        http.add_header("anthropic-version", options_.api_version);
        http.add_header("content-type", "application/json");
        http.add_header("accept", "text/event-stream");
        if (!options_.betas.empty()) {
            std::string joined;
            for (const std::string& b : options_.betas) {
                joined += (joined.empty() ? "" : ",") + b;
            }
            http.add_header("anthropic-beta", joined);
        }
        http.body = WriteJsonCompact(build_body(request));

        return detail::WithRetries(options_.retry, rng_, "anthropic", [&](bool& streamed) {
            StreamAssembler assembler(stream, streamed);
            std::exception_ptr failure;
            SseParser parser([&](const SseEvent& e) { assembler.on_event(e); });
            HttpResponse response = transport_->stream(http, detail::Guarded(failure, [&](std::string_view chunk) {
                parser.feed(chunk);
            }));
            if (failure) {
                std::rethrow_exception(failure);
            }
            if (!response.ok()) {
                detail::ThrowForStatus(response, detail::ErrorDialect::Anthropic, "anthropic");
            }
            parser.finish();
            const std::string* id = response.header("request-id");
            return assembler.finish(id != nullptr ? *id : "");
        });
    }

    std::string name() const override { return "anthropic (" + options_.base_url + ")"; }

private:
    JsonValue build_body(const ChatRequest& request) const {
        JsonValue body = EmptyObject();
        body.add("model", request.model);
        body.add("max_tokens", request.max_tokens);
        if (request.temperature) {
            body.add("temperature", *request.temperature);
        }
        body.add("stream", true);
        if (options_.prompt_caching) {
            body.add("cache_control", CacheControl());  // automatic: the conversation so far
        }
        if (!request.system.empty()) {
            JsonValue block = EmptyObject();
            block.add("type", "text");
            block.add("text", request.system);
            if (options_.prompt_caching) {
                block.add("cache_control", CacheControl());
            }
            JsonValue system = EmptyArray();
            system.push_back(std::move(block));
            body.add("system", std::move(system));
        }
        if (!request.tools.empty()) {
            JsonValue tools = EmptyArray();
            for (std::size_t i = 0; i < request.tools.size(); ++i) {
                const ToolSpec& t = request.tools[i];
                JsonValue tool = EmptyObject();
                tool.add("name", t.name);
                tool.add("description", t.description);
                tool.add("input_schema", t.input_schema);
                if (t.strict) {
                    tool.add("strict", true);
                }
                if (options_.prompt_caching && i + 1 == request.tools.size()) {
                    tool.add("cache_control", CacheControl());  // caches every tool before it too
                }
                tools.push_back(std::move(tool));
            }
            body.add("tools", std::move(tools));
            if (!request.allow_tool_calls) {
                JsonValue choice = EmptyObject();
                choice.add("type", "none");
                body.add("tool_choice", std::move(choice));
            }
            // Otherwise no tool_choice: "auto" is the default, and "any" or "tool" return HTTP 400
            // on current models.
        }
        JsonValue messages = EmptyArray();
        for (const Message& m : request.messages) {
            JsonValue content = EmptyArray();
            for (const ContentBlock& b : m.content) {
                if (Sendable(b)) {
                    content.push_back(BlockToJson(b));
                }
            }
            JsonValue message = EmptyObject();
            message.add("role", m.role == Message::Role::User ? "user" : "assistant");
            message.add("content", std::move(content));
            messages.push_back(std::move(message));
        }
        body.add("messages", std::move(messages));
        return body;
    }

    AnthropicOptions options_;
    std::shared_ptr<HttpTransport> transport_;
    std::mt19937_64 rng_;
};

}  // namespace

std::unique_ptr<Provider> MakeAnthropicProvider(AnthropicOptions options, std::shared_ptr<HttpTransport> transport) {
    return std::make_unique<AnthropicProvider>(std::move(options), std::move(transport));
}

}  // namespace pulsatrix::orch
