/** @file tool_loop.hpp
 *  @brief The agent loop: ask the model, run the tools it calls, send back the results, repeat.
 *  @ingroup orchestrator
 */
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "pulsatrix/orch/message.hpp"
#include "pulsatrix/orch/provider.hpp"

namespace pulsatrix::orch {

/** @brief One tool call to run. */
struct ToolCall {
    std::string id;
    std::string name;
    JsonValue input;
};

/** @brief What a tool returned. An error is reported to the model, which can try again. */
struct ToolOutcome {
    std::string content;
    bool is_error = false;
};

/**
 * @brief Runs one tool call. AGT-4's tool registry provides one; any function works.
 * @note An exception is caught and sent to the model as an error result with its message.
 */
using ToolExecutor = std::function<ToolOutcome(const ToolCall&)>;

/** @brief Settings for RunToolLoop(). */
struct ToolLoopOptions {
    /** @brief The most requests to make. The loop stops there even if the model wants more tools. */
    int max_turns = 20;
    StreamCallbacks stream;
    /** @brief Called with each message the loop adds: the model's answers and the tool results. */
    std::function<void(const Message&)> on_message;
};

/** @brief How a tool loop ended. */
struct ToolLoopResult {
    /** @brief The whole conversation: the request's messages, then everything the loop added. */
    std::vector<Message> messages;
    /** @brief The last answer. */
    ChatResponse last;
    int turns = 0;
    /** @brief Usage summed over every request. */
    Usage usage;
    /** @brief True if the loop stopped at max_turns while the model still wanted tools. */
    bool hit_turn_limit = false;
};

/**
 * @brief Sends @p request, runs each tool the model calls with @p execute, and sends the results
 *        back, until the model stops calling tools or @p options.max_turns requests have been made.
 * @note The results of one answer's calls go back together, first in the next user message and
 *       in the order of the calls, as the Anthropic API requires. A PauseTurn answer is sent back
 *       as it is, to let the model continue.
 * @throws ProviderError or HttpError as Provider::chat() does.
 */
[[nodiscard]] ToolLoopResult RunToolLoop(Provider& provider, ChatRequest request, const ToolExecutor& execute,
                                         const ToolLoopOptions& options = {});

}  // namespace pulsatrix::orch
