// OpenAI-compatible chat completions (POST <base>/chat/completions), always streamed: what
// llama-server, vLLM and Ollama serve. Tool calls arrive as fragments per call index; thinking,
// when a server streams it as reasoning_content, is shown but never sent back.
#include <exception>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "provider_internal.hpp"
#include "pulsatrix/orch/provider.hpp"
#include "pulsatrix/orch/sse.hpp"

namespace pulsatrix::orch {

namespace {

using detail::EmptyArray;
using detail::EmptyObject;
using detail::IntOr;
using detail::ObjectMember;
using detail::StringOr;

constexpr const char* kOrigin = "openai-compatible";

StopReason ParseFinishReason(const std::string& s) {
    if (s == "stop") return StopReason::EndTurn;
    if (s == "tool_calls" || s == "function_call") return StopReason::ToolUse;
    if (s == "length") return StopReason::MaxTokens;
    if (s == "content_filter") return StopReason::Refusal;
    return StopReason::Other;
}

class ChunkAssembler {
public:
    ChunkAssembler(const StreamCallbacks& callbacks, bool& streamed) : callbacks_(callbacks), streamed_(streamed) {}

    void on_event(const SseEvent& event) {
        if (event.data == "[DONE]") {
            done_ = true;
            return;
        }
        JsonValue data;
        try {
            data = ParseJson(event.data);
        } catch (const std::invalid_argument& e) {
            throw HttpError(HttpError::Kind::Protocol, std::string("openai-compatible: invalid JSON in a stream chunk: ") + e.what());
        }
        if (data.type() != JsonValue::Type::Object) {
            return;
        }
        if (const JsonValue* error = data.find("error")) {
            const std::string message = error->type() == JsonValue::Type::Object ? StringOr(*error, "message") : event.data;
            throw ProviderError("openai-compatible: error in the stream: " + message, 0, StringOr(*error, "type"), "");
        }
        model_ = StringOr(data, "model", model_);
        if (const JsonValue* usage = ObjectMember(data, "usage")) {
            usage_.input_tokens = IntOr(*usage, "prompt_tokens", usage_.input_tokens);
            usage_.output_tokens = IntOr(*usage, "completion_tokens", usage_.output_tokens);
            if (const JsonValue* details = ObjectMember(*usage, "prompt_tokens_details")) {
                usage_.cache_read_input_tokens = IntOr(*details, "cached_tokens", usage_.cache_read_input_tokens);
            }
        }
        const JsonValue* choices = data.find("choices");
        if (choices == nullptr || choices->type() != JsonValue::Type::Array || choices->as_array().empty()) {
            return;  // the usage chunk has no choices
        }
        const JsonValue& choice = choices->as_array()[0];
        const std::string finish = StringOr(choice, "finish_reason");
        if (!finish.empty()) {
            finish_reason_ = finish;
        }
        const JsonValue* delta = ObjectMember(choice, "delta");
        if (delta == nullptr) {
            return;
        }
        for (const char* key : {"reasoning_content", "reasoning"}) {
            const std::string thinking = StringOr(*delta, key);
            if (!thinking.empty()) {
                thinking_ += thinking;
                if (callbacks_.on_thinking) {
                    streamed_ = true;
                    callbacks_.on_thinking(thinking);
                }
            }
        }
        const std::string text = StringOr(*delta, "content");
        if (!text.empty()) {
            text_ += text;
            if (callbacks_.on_text) {
                streamed_ = true;
                callbacks_.on_text(text);
            }
        }
        if (const JsonValue* calls = delta->find("tool_calls"); calls != nullptr && calls->type() == JsonValue::Type::Array) {
            for (const JsonValue& call : calls->as_array()) {
                Call& c = calls_[IntOr(call, "index", static_cast<int64_t>(calls_.size()))];
                c.id = StringOr(call, "id", c.id);
                if (const JsonValue* function = ObjectMember(call, "function")) {
                    c.name += StringOr(*function, "name");
                    c.arguments += StringOr(*function, "arguments");
                }
            }
        }
    }

    ChatResponse finish() {
        if (!done_ && finish_reason_.empty()) {
            throw HttpError(HttpError::Kind::Protocol, "openai-compatible: the stream ended before a finish_reason");
        }
        ChatResponse response;
        response.message.role = Message::Role::Assistant;
        if (!thinking_.empty()) {
            ContentBlock thinking;
            thinking.type = ContentBlock::Type::Thinking;
            thinking.text = thinking_;
            thinking.origin = kOrigin;  // shown, never sent back
            response.message.content.push_back(std::move(thinking));
        }
        if (!text_.empty()) {
            response.message.content.push_back(ContentBlock::Text(text_));
        }
        int n = 0;
        for (auto& [index, call] : calls_) {
            // Some servers (Ollama, for one) leave out the id.
            std::string id = call.id.empty() ? "call_" + std::to_string(n) : call.id;
            ++n;
            response.message.content.push_back(ContentBlock::ToolUse(std::move(id), call.name, parse_arguments(call.arguments)));
        }
        response.stop_reason_text = finish_reason_;
        response.stop_reason = ParseFinishReason(finish_reason_);
        if (!calls_.empty() && response.stop_reason == StopReason::EndTurn) {
            response.stop_reason = StopReason::ToolUse;  // some servers say "stop" after tool calls
        }
        response.usage = usage_;
        response.model = model_;
        return response;
    }

private:
    struct Call {
        std::string id;
        std::string name;
        std::string arguments;
    };

    // A small model may write arguments that aren't JSON. They're kept as a string, so the tool
    // (or AGT-4's validation) reports the error back to the model instead of the loop failing.
    static JsonValue parse_arguments(const std::string& arguments) {
        if (arguments.empty()) {
            return EmptyObject();
        }
        try {
            return ParseJson(arguments);
        } catch (const std::invalid_argument&) {
            return JsonValue(arguments);
        }
    }

    const StreamCallbacks& callbacks_;
    bool& streamed_;
    std::string text_;
    std::string thinking_;
    std::map<int64_t, Call> calls_;
    std::string finish_reason_;
    std::string model_;
    Usage usage_;
    bool done_ = false;
};

class OpenAiCompatibleProvider final : public Provider {
public:
    OpenAiCompatibleProvider(OpenAiCompatibleOptions options, std::shared_ptr<HttpTransport> transport)
        : options_(std::move(options)), transport_(std::move(transport)), rng_(options_.retry.seed) {
        if (!transport_) {
            transport_ = MakeHttpTransport();
        }
    }

    ChatResponse chat(const ChatRequest& request, const StreamCallbacks& stream) override {
        detail::ValidateChatRequest(request);
        HttpRequest http;
        http.url = options_.base_url + "/chat/completions";
        http.read_timeout = options_.read_timeout;
        if (options_.api_key) {
            http.add_secret_header("authorization", Secret("Bearer " + options_.api_key->reveal()));
        }
        http.add_header("content-type", "application/json");
        http.add_header("accept", "text/event-stream");
        http.body = WriteJsonCompact(build_body(request));

        return detail::WithRetries(options_.retry, rng_, "openai-compatible", [&](bool& streamed) {
            ChunkAssembler assembler(stream, streamed);
            std::exception_ptr failure;
            SseParser parser([&](const SseEvent& e) { assembler.on_event(e); });
            HttpResponse response = transport_->stream(http, detail::Guarded(failure, [&](std::string_view chunk) {
                parser.feed(chunk);
            }));
            if (failure) {
                std::rethrow_exception(failure);
            }
            if (!response.ok()) {
                detail::ThrowForStatus(response, detail::ErrorDialect::OpenAi, "openai-compatible");
            }
            parser.finish();
            return assembler.finish();
        });
    }

    std::string name() const override { return "openai-compatible (" + options_.base_url + ")"; }

private:
    static JsonValue build_body(const ChatRequest& request) {
        JsonValue body = EmptyObject();
        body.add("model", request.model);
        body.add("max_tokens", request.max_tokens);
        if (request.temperature) {
            body.add("temperature", *request.temperature);
        }
        body.add("stream", true);
        JsonValue stream_options = EmptyObject();
        stream_options.add("include_usage", true);
        body.add("stream_options", std::move(stream_options));

        JsonValue messages = EmptyArray();
        if (!request.system.empty()) {
            JsonValue system = EmptyObject();
            system.add("role", "system");
            system.add("content", request.system);
            messages.push_back(std::move(system));
        }
        for (const Message& m : request.messages) {
            if (m.role == Message::Role::User) {
                // Tool results become "tool" messages, which must directly follow the assistant
                // message with the calls; any text goes in a user message after them.
                std::string text;
                for (const ContentBlock& b : m.content) {
                    if (b.type == ContentBlock::Type::ToolResult) {
                        JsonValue tool = EmptyObject();
                        tool.add("role", "tool");
                        tool.add("tool_call_id", b.id);
                        tool.add("content", b.is_error ? "Error: " + b.text : b.text);
                        messages.push_back(std::move(tool));
                    } else if (b.type == ContentBlock::Type::Text) {
                        text += b.text;
                    }
                }
                if (!text.empty()) {
                    JsonValue user = EmptyObject();
                    user.add("role", "user");
                    user.add("content", text);
                    messages.push_back(std::move(user));
                }
                continue;
            }
            JsonValue assistant = EmptyObject();
            assistant.add("role", "assistant");
            const std::string text = m.text();
            assistant.add("content", text.empty() ? JsonValue() : JsonValue(text));
            JsonValue calls = EmptyArray();
            for (const ContentBlock* call : m.tool_uses()) {
                JsonValue function = EmptyObject();
                function.add("name", call->name);
                function.add("arguments", call->input.type() == JsonValue::Type::String ? call->input.as_string()
                                                                                      : WriteJsonCompact(call->input));
                JsonValue c = EmptyObject();
                c.add("id", call->id);
                c.add("type", "function");
                c.add("function", std::move(function));
                calls.push_back(std::move(c));
            }
            if (!calls.as_array().empty()) {
                assistant.add("tool_calls", std::move(calls));
            }
            messages.push_back(std::move(assistant));
        }
        body.add("messages", std::move(messages));

        if (!request.tools.empty()) {
            JsonValue tools = EmptyArray();
            for (const ToolSpec& t : request.tools) {
                JsonValue function = EmptyObject();
                function.add("name", t.name);
                function.add("description", t.description);
                function.add("parameters", t.input_schema);
                if (t.strict) {
                    function.add("strict", true);
                }
                JsonValue tool = EmptyObject();
                tool.add("type", "function");
                tool.add("function", std::move(function));
                tools.push_back(std::move(tool));
            }
            body.add("tools", std::move(tools));
            body.add("tool_choice", request.allow_tool_calls ? "auto" : "none");
        }
        return body;
    }

    OpenAiCompatibleOptions options_;
    std::shared_ptr<HttpTransport> transport_;
    std::mt19937_64 rng_;
};

}  // namespace

std::unique_ptr<Provider> MakeOpenAiCompatibleProvider(OpenAiCompatibleOptions options, std::shared_ptr<HttpTransport> transport) {
    return std::make_unique<OpenAiCompatibleProvider>(std::move(options), std::move(transport));
}

}  // namespace pulsatrix::orch
