#include "pulsatrix/orch/tool_registry.hpp"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/orch/json_schema.hpp"

namespace pulsatrix::orch {

ToolRegistry& ToolRegistry::add(Tool tool) {
    if (!IsValidToolName(tool.name)) {
        throw std::invalid_argument("ToolRegistry: a tool name must be 1 to 128 letters, digits, '_' or '-'");
    }
    const std::string prefix = "ToolRegistry: tool \"" + tool.name + "\": ";
    if (index_.count(tool.name) != 0) {
        throw std::invalid_argument(prefix + "there is already a tool with this name");
    }
    if (!tool.handler) {
        throw std::invalid_argument(prefix + "no handler");
    }
    const JsonValue* type = tool.input_schema.type() == JsonValue::Type::Object ? tool.input_schema.find("type") : nullptr;
    if (type == nullptr || type->type() != JsonValue::Type::String || type->as_string() != "object") {
        throw std::invalid_argument(prefix + "the input schema must be an object with \"type\": \"object\"");
    }
    try {
        CheckSchema(tool.input_schema);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(prefix + e.what());
    }
    if (tool.strict) {
        const std::string problem = StrictModeProblem(tool.input_schema);
        if (!problem.empty()) {
            throw std::invalid_argument(prefix + "it's strict, but " + problem);
        }
    }
    index_.emplace(tool.name, tools_.size());
    tools_.push_back(std::move(tool));
    return *this;
}

const Tool* ToolRegistry::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tools_[it->second];
}

std::vector<ToolSpec> ToolRegistry::specs() const {
    std::vector<ToolSpec> out;
    out.reserve(tools_.size());
    for (const Tool& t : tools_) {
        out.push_back({t.name, t.description, t.input_schema, t.strict});
    }
    return out;
}

ToolOutcome ToolRegistry::call(std::string_view name, const JsonValue& input) const {
    const Tool* tool = find(name);
    if (tool == nullptr) {
        // The name came from the model, so it's only repeated if it's a valid one.
        std::string message = IsValidToolName(std::string(name)) ? "There is no tool named \"" + std::string(name) + "\"."
                                                                 : "There is no tool with that name.";
        message += " The tools are:";
        for (std::size_t i = 0; i < tools_.size(); ++i) {
            message += (i == 0 ? " " : ", ") + tools_[i].name;
        }
        message += tools_.empty() ? " none." : ".";
        return {message, true};
    }
    if (input.type() == JsonValue::Type::String) {
        // A provider keeps arguments that weren't JSON as a string.
        return {"The input for " + tool->name + " wasn't valid JSON. Send a JSON object that matches its schema.", true};
    }
    const std::vector<SchemaError> errors = ValidateJson(tool->input_schema, input);
    if (!errors.empty()) {
        std::string message = "The input for " + tool->name + " doesn't match its schema:";
        for (const SchemaError& e : errors) {
            message += "\n- " + (e.path.empty() ? std::string("the input") : e.path) + " " + e.message;
        }
        return {message, true};
    }
    try {
        return tool->handler(input);
    } catch (const std::exception& e) {
        return {std::string("The tool failed: ") + e.what(), true};
    } catch (...) {
        return {"The tool failed with an unknown error.", true};
    }
}

ToolExecutor ToolRegistry::executor() const {
    return [this](const ToolCall& c) { return call(c.name, c.input); };
}

}  // namespace pulsatrix::orch
