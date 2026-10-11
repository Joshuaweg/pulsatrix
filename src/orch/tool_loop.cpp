#include "pulsatrix/orch/tool_loop.hpp"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsatrix::orch {

namespace {

ToolOutcome RunOne(const ToolExecutor& execute, const ToolCall& call) {
    try {
        return execute(call);
    } catch (const std::exception& e) {
        return {std::string("The tool failed: ") + e.what(), true};
    } catch (...) {
        return {"The tool failed with an unknown error.", true};
    }
}

}  // namespace

ToolLoopResult RunToolLoop(Provider& provider, ChatRequest request, const ToolExecutor& execute, const ToolLoopOptions& options) {
    if (options.max_turns < 1) {
        throw std::invalid_argument("ToolLoopOptions: max_turns must be at least 1");
    }
    if (!execute) {
        throw std::invalid_argument("RunToolLoop: no tool executor");
    }
    ToolLoopResult result;
    while (true) {
        ChatResponse response = provider.chat(request, options.stream);
        ++result.turns;
        result.usage += response.usage;
        request.messages.push_back(response.message);
        if (options.on_message) {
            options.on_message(response.message);
        }
        const StopReason stop = response.stop_reason;
        const bool wants_tools = stop == StopReason::ToolUse && !response.message.tool_uses().empty();
        result.last = std::move(response);

        if (stop == StopReason::PauseTurn) {
            // The provider paused a long turn; sending the conversation back lets it continue.
            if (result.turns >= options.max_turns) {
                result.hit_turn_limit = true;
                break;
            }
            continue;
        }
        if (!wants_tools) {
            break;
        }

        // Every call's result, in the calls' order, first in the next user message.
        Message results;
        results.role = Message::Role::User;
        for (const ContentBlock* call : request.messages.back().tool_uses()) {
            const ToolOutcome outcome = RunOne(execute, {call->id, call->name, call->input});
            results.content.push_back(ContentBlock::ToolResult(call->id, outcome.content, outcome.is_error));
        }
        request.messages.push_back(results);
        if (options.on_message) {
            options.on_message(results);
        }
        if (result.turns >= options.max_turns) {
            result.hit_turn_limit = true;
            break;
        }
    }
    result.messages = std::move(request.messages);
    return result;
}

}  // namespace pulsatrix::orch
