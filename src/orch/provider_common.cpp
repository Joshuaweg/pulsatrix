#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <utility>

#include "provider_internal.hpp"
#include "pulsatrix/orch/message.hpp"

namespace pulsatrix::orch {

// ---- message.hpp ---------------------------------------------------------------------------

ContentBlock ContentBlock::Text(std::string text) {
    ContentBlock b;
    b.type = Type::Text;
    b.text = std::move(text);
    return b;
}

ContentBlock ContentBlock::ToolUse(std::string id, std::string name, JsonValue input) {
    ContentBlock b;
    b.type = Type::ToolUse;
    b.id = std::move(id);
    b.name = std::move(name);
    b.input = std::move(input);
    return b;
}

ContentBlock ContentBlock::ToolResult(std::string tool_use_id, std::string content, bool is_error) {
    ContentBlock b;
    b.type = Type::ToolResult;
    b.id = std::move(tool_use_id);
    b.text = std::move(content);
    b.is_error = is_error;
    return b;
}

Message Message::User(std::string text) {
    Message m;
    m.role = Role::User;
    m.content.push_back(ContentBlock::Text(std::move(text)));
    return m;
}

Message Message::Assistant(std::string text) {
    Message m;
    m.role = Role::Assistant;
    m.content.push_back(ContentBlock::Text(std::move(text)));
    return m;
}

std::string Message::text() const {
    std::string out;
    for (const ContentBlock& b : content) {
        if (b.type == ContentBlock::Type::Text) {
            out += b.text;
        }
    }
    return out;
}

std::vector<const ContentBlock*> Message::tool_uses() const {
    std::vector<const ContentBlock*> out;
    for (const ContentBlock& b : content) {
        if (b.type == ContentBlock::Type::ToolUse) {
            out.push_back(&b);
        }
    }
    return out;
}

Usage& Usage::operator+=(const Usage& other) {
    input_tokens += other.input_tokens;
    output_tokens += other.output_tokens;
    cache_creation_input_tokens += other.cache_creation_input_tokens;
    cache_read_input_tokens += other.cache_read_input_tokens;
    return *this;
}

bool IsValidToolName(const std::string& name) {
    if (name.empty() || name.size() > 128) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}

namespace detail {

// ---- JSON helpers --------------------------------------------------------------------------

JsonValue EmptyObject() { return JsonValue{JsonValue::Object{}}; }
JsonValue EmptyArray() { return JsonValue{JsonValue::Array{}}; }

std::string StringOr(const JsonValue& object, std::string_view key, std::string fallback) {
    if (object.type() != JsonValue::Type::Object) {
        return fallback;
    }
    const JsonValue* v = object.find(key);
    return v != nullptr && v->type() == JsonValue::Type::String ? v->as_string() : fallback;
}

int64_t IntOr(const JsonValue& object, std::string_view key, int64_t fallback) {
    if (object.type() != JsonValue::Type::Object) {
        return fallback;
    }
    const JsonValue* v = object.find(key);
    if (v == nullptr || v->type() != JsonValue::Type::Number) {
        return fallback;
    }
    try {
        return v->as_int64();
    } catch (const std::invalid_argument&) {
        return fallback;
    }
}

const JsonValue* ObjectMember(const JsonValue& object, std::string_view key) {
    if (object.type() != JsonValue::Type::Object) {
        return nullptr;
    }
    const JsonValue* v = object.find(key);
    return v != nullptr && v->type() == JsonValue::Type::Object ? v : nullptr;
}

JsonValue WithMember(const JsonValue& object, const std::string& key, JsonValue value) {
    JsonValue out = EmptyObject();
    for (const auto& [k, v] : object.as_object()) {
        if (k != key) {
            out.add(k, v);
        }
    }
    out.add(key, std::move(value));
    return out;
}

// ---- errors and retries --------------------------------------------------------------------

namespace {

std::optional<std::chrono::milliseconds> RetryAfter(const HttpResponse& response) {
    const std::string* value = response.header("retry-after");
    if (value == nullptr) {
        return std::nullopt;
    }
    // Seconds, possibly fractional. The HTTP-date form isn't used by these APIs; an unreadable
    // value counts as present but unknown, so the backoff decides.
    try {
        std::size_t used = 0;
        const double seconds = std::stod(*value, &used);
        if (used == value->size() && seconds >= 0 && std::isfinite(seconds)) {
            return std::chrono::milliseconds(static_cast<int64_t>(std::ceil(seconds * 1000)));
        }
    } catch (const std::exception&) {
    }
    return std::chrono::milliseconds(-1);
}

}  // namespace

void ThrowForStatus(const HttpResponse& response, ErrorDialect dialect, const std::string& provider) {
    std::string type;
    std::string message;
    std::string request_id;
    try {
        const JsonValue body = ParseJson(response.body);
        if (const JsonValue* error = ObjectMember(body, "error")) {
            type = StringOr(*error, "type");
            message = StringOr(*error, "message");
            if (const JsonValue* details = ObjectMember(*error, "details")) {
                const std::string code = StringOr(*details, "error_code");
                if (!code.empty()) {
                    message += " [" + code + "]";
                }
            }
        } else {
            message = StringOr(body, "error", StringOr(body, "message"));
        }
        request_id = StringOr(body, "request_id");
    } catch (const std::exception&) {
        message = response.body.substr(0, 500);  // not JSON: an HTML error page from a proxy
    }
    if (request_id.empty()) {
        if (const std::string* id = response.header("request-id")) {
            request_id = *id;
        }
    }
    std::string text = provider + ": HTTP " + std::to_string(response.status);
    if (!type.empty()) {
        text += " " + type;
    }
    if (!message.empty()) {
        text += ": " + message;
    }
    if (!request_id.empty()) {
        text += " (request " + request_id + ")";
    }
    ProviderError error(text, response.status, type, request_id);

    const int s = response.status;
    const std::optional<std::chrono::milliseconds> retry_after = RetryAfter(response);
    if (s == 429) {
        if (!retry_after && dialect == ErrorDialect::Anthropic) {
            // The monthly spend cap: no retry-after, and every retry fails until it resets.
            throw ProviderError(text + "; no retry-after, so not retried (a spend cap, not a rate limit)", s, type,
                                request_id);
        }
        throw RetryableFailure{error, retry_after};
    }
    if (s == 408 || s == 500 || s == 502 || s == 503 || s == 504 || s == 529) {
        throw RetryableFailure{error, retry_after};
    }
    throw error;
}

ChatResponse WithRetries(const RetryPolicy& policy, std::mt19937_64& rng, const std::string& provider,
                         const std::function<ChatResponse(bool& streamed)>& attempt) {
    for (int retry = 0;; ++retry) {
        bool streamed = false;
        std::optional<RetryableFailure> failure;
        try {
            return attempt(streamed);
        } catch (const RetryableFailure& f) {
            failure = f;
        } catch (const HttpError& e) {
            if (e.kind() != HttpError::Kind::Connect && e.kind() != HttpError::Kind::Timeout &&
                e.kind() != HttpError::Kind::Protocol) {
                throw;  // a bad request, no TLS, a body over the limit: retrying can't help
            }
            failure = RetryableFailure{ProviderError(provider + ": " + e.what(), 0, "transport", ""), std::nullopt};
        }
        if (streamed) {
            throw ProviderError(std::string(failure->error.what()) + "; not retried, since part of the answer was already streamed",
                                failure->error.status(), failure->error.type(), failure->error.request_id());
        }
        if (retry >= policy.max_retries) {
            throw failure->error;
        }
        std::chrono::milliseconds wait{0};
        if (failure->retry_after && failure->retry_after->count() >= 0) {
            if (*failure->retry_after > policy.max_retry_after) {
                throw ProviderError(std::string(failure->error.what()) + "; retry-after is longer than max_retry_after",
                                    failure->error.status(), failure->error.type(), failure->error.request_id());
            }
            wait = *failure->retry_after;
        } else {
            const double cap = std::min<double>(static_cast<double>(policy.max_backoff.count()),
                                                static_cast<double>(policy.initial_backoff.count()) * std::pow(2.0, retry));
            wait = std::chrono::milliseconds(static_cast<int64_t>(std::uniform_real_distribution<double>(0.0, cap)(rng)));
        }
        if (policy.sleep) {
            policy.sleep(wait);
        } else {
            std::this_thread::sleep_for(wait);
        }
    }
}

HttpChunkHandler Guarded(std::exception_ptr& failure, std::function<void(std::string_view)> handler) {
    return [&failure, handler = std::move(handler)](std::string_view chunk) {
        try {
            handler(chunk);
            return true;
        } catch (...) {
            failure = std::current_exception();
            return false;
        }
    };
}

void ValidateChatRequest(const ChatRequest& request) {
    if (request.model.empty()) {
        throw std::invalid_argument("ChatRequest: model is empty");
    }
    if (request.messages.empty()) {
        throw std::invalid_argument("ChatRequest: no messages");
    }
    if (request.max_tokens < 1) {
        throw std::invalid_argument("ChatRequest: max_tokens must be at least 1");
    }
    for (const ToolSpec& tool : request.tools) {
        if (!IsValidToolName(tool.name)) {
            throw std::invalid_argument("ChatRequest: tool name '" + tool.name + "' must match ^[a-zA-Z0-9_-]{1,128}$");
        }
        if (tool.input_schema.type() != JsonValue::Type::Object) {
            throw std::invalid_argument("ChatRequest: tool '" + tool.name + "' has no input_schema object");
        }
    }
}

}  // namespace detail
}  // namespace pulsatrix::orch
