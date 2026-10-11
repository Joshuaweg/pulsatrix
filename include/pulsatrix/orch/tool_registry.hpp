/** @file tool_registry.hpp
 *  @brief Tools defined once: their specs for a provider, and an executor that checks inputs.
 *  @ingroup orchestrator
 */
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/orch/message.hpp"
#include "pulsatrix/orch/tool_loop.hpp"

namespace pulsatrix::orch {

/** @brief Runs a tool on an input that already matches its schema. An exception becomes an
 *         error result with the exception's message. */
using ToolHandler = std::function<ToolOutcome(const JsonValue& input)>;

/** @brief A tool: what the model sees, and the function that runs it. */
struct Tool {
    std::string name;  ///< letters, digits, `_` and `-`, 1 to 128 characters
    std::string description;
    /** @brief A JSON Schema with `"type": "object"`, using only the keywords json_schema.hpp lists. */
    JsonValue input_schema;
    ToolHandler handler;
    /** @brief Ask the provider to guarantee inputs that match the schema. The schema must suit
     *         strict mode (see StrictModeProblem()). Inputs are checked here either way. */
    bool strict = false;
};

/**
 * @brief The tools an agent may use. RunToolLoop() takes executor() and the MCP server (AGT-5)
 *        serves the same registry, so a tool is defined once.
 * @note Calls are const and may run concurrently if the handlers allow it; add() may not.
 */
class ToolRegistry {
public:
    /**
     * @brief Adds @p tool.
     * @throws std::invalid_argument if its name is invalid or taken, it has no handler, its
     *         schema isn't an object schema that passes CheckSchema(), or it's strict and the
     *         schema doesn't suit strict mode.
     */
    ToolRegistry& add(Tool tool);

    /** @brief The tool named @p name, or nullptr. */
    [[nodiscard]] const Tool* find(std::string_view name) const;
    [[nodiscard]] std::size_t size() const { return tools_.size(); }
    [[nodiscard]] bool empty() const { return tools_.empty(); }
    /** @brief The tools, in the order they were added. */
    [[nodiscard]] const std::vector<Tool>& tools() const { return tools_; }

    /** @brief The tools as ChatRequest::tools wants them, in the order they were added. */
    [[nodiscard]] std::vector<ToolSpec> specs() const;

    /**
     * @brief Runs tool @p name on @p input. An unknown tool, an input that doesn't match the
     *        schema, or a handler that throws gives an error result the model can act on,
     *        never an exception.
     */
    [[nodiscard]] ToolOutcome call(std::string_view name, const JsonValue& input) const;

    /** @brief An executor for RunToolLoop() that calls call(). It refers to this registry, which
     *         must outlive it. */
    [[nodiscard]] ToolExecutor executor() const;

private:
    std::vector<Tool> tools_;
    std::map<std::string, std::size_t, std::less<>> index_;
};

}  // namespace pulsatrix::orch
