/** @file message.hpp
 *  @brief The orchestrator's message format, shared by every LLM provider.
 *  @ingroup orchestrator
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"

namespace pulsatrix::orch {

/** @brief One block of a message's content. */
struct ContentBlock {
    enum class Type {
        Text,
        ToolUse,     ///< the model calls a tool: id, name and input
        ToolResult,  ///< the answer to a ToolUse: id (the call's), text and is_error
        Thinking,    ///< the model's reasoning: text, plus the provider's block in raw
        Opaque,      ///< anything else the provider sent: raw only
    };

    Type type = Type::Text;
    /** @brief Text, a tool result's content, or thinking. */
    std::string text;
    /** @brief A tool call's id, or for a result the id of the call it answers. */
    std::string id;
    /** @brief A tool call's tool name. */
    std::string name;
    /** @brief A tool call's input, a JSON object. */
    JsonValue input;
    /** @brief Whether a tool result reports a failure. */
    bool is_error = false;
    /**
     * @brief For Thinking and Opaque blocks, the block exactly as the provider sent it, and the
     *        provider's name in @ref origin. The provider sends it back unchanged: the Anthropic
     *        API rejects a request whose thinking blocks were altered. Other providers skip it.
     */
    JsonValue raw;
    std::string origin;

    [[nodiscard]] static ContentBlock Text(std::string text);
    [[nodiscard]] static ContentBlock ToolUse(std::string id, std::string name, JsonValue input);
    [[nodiscard]] static ContentBlock ToolResult(std::string tool_use_id, std::string content, bool is_error = false);
};

/** @brief One turn of a conversation. */
struct Message {
    enum class Role { User, Assistant };

    Role role = Role::User;
    std::vector<ContentBlock> content;

    [[nodiscard]] static Message User(std::string text);
    [[nodiscard]] static Message Assistant(std::string text);

    /** @brief The text blocks joined, without thinking or tool calls. */
    [[nodiscard]] std::string text() const;
    /** @brief The tool calls, in order. */
    [[nodiscard]] std::vector<const ContentBlock*> tool_uses() const;
};

/** @brief A tool the model may call. ToolRegistry::specs() makes these from its tools. */
struct ToolSpec {
    std::string name;  ///< letters, digits, `_` and `-`, 1 to 128 characters
    std::string description;
    JsonValue input_schema;  ///< a JSON Schema object
    /** @brief Ask the provider to guarantee inputs that match the schema (Anthropic's strict tool
     *         use). The schema must then follow the provider's strict-mode rules, such as
     *         `"additionalProperties": false` on every object. */
    bool strict = false;
};

/** @brief Why the model stopped. */
enum class StopReason {
    EndTurn,       ///< it finished its answer
    ToolUse,       ///< it called tools and waits for their results
    MaxTokens,     ///< it reached max_tokens
    StopSequence,  ///< it produced a stop sequence
    PauseTurn,     ///< the provider paused a long turn; send the conversation back to continue
    Refusal,       ///< it declined
    Other,         ///< a reason this version doesn't know; see ChatResponse::stop_reason_text
};

/** @brief Tokens used by one request, or summed over a tool loop. */
struct Usage {
    int64_t input_tokens = 0;
    int64_t output_tokens = 0;
    int64_t cache_creation_input_tokens = 0;
    int64_t cache_read_input_tokens = 0;

    Usage& operator+=(const Usage& other);
};

/** @brief A request to a provider. */
struct ChatRequest {
    std::string model;
    std::string system;
    std::vector<Message> messages;
    std::vector<ToolSpec> tools;
    int64_t max_tokens = 4096;
    std::optional<double> temperature;
    /** @brief Whether the model may call the tools. False sends them anyway, for the cache, and
     *         tells the model not to call any. A specific tool can't be forced: current Claude
     *         models reject that with HTTP 400. */
    bool allow_tool_calls = true;
};

/** @brief A provider's answer. */
struct ChatResponse {
    Message message;  ///< always an assistant message
    StopReason stop_reason = StopReason::EndTurn;
    std::string stop_reason_text;  ///< the provider's own word for it, for example `end_turn`
    Usage usage;
    std::string model;       ///< the model that answered, as the provider reports it
    std::string request_id;  ///< for a support ticket, when the provider sends one
};

/** @brief Whether @p name is a valid tool name (`^[a-zA-Z0-9_-]{1,128}$`). */
[[nodiscard]] bool IsValidToolName(const std::string& name);

}  // namespace pulsatrix::orch
