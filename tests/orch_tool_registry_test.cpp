// AGT-4: the JSON Schema validator and the tool registry, and the registry driving RunToolLoop()
// through a scripted provider.
#include "pulsatrix/orch/tool_registry.hpp"

#include <gtest/gtest.h>

#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/orch/json_schema.hpp"
#include "pulsatrix/orch/provider.hpp"
#include "pulsatrix/orch/tool_loop.hpp"

namespace pulsatrix::orch {
namespace {

JsonValue J(const char* text) { return ParseJson(text); }

std::vector<std::string> Messages(const std::vector<SchemaError>& errors) {
    std::vector<std::string> out;
    for (const SchemaError& e : errors) {
        out.push_back(e.path + " " + e.message);
    }
    return out;
}

bool Valid(const char* schema, const char* value) { return ValidateJson(J(schema), J(value)).empty(); }

// The thrown message, or "" if nothing was thrown.
std::string CheckError(const char* schema) {
    try {
        CheckSchema(J(schema));
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

// ---- checking schemas ----------------------------------------------------------------------

TEST(OrchJsonSchemaTest, AcceptsTheSharedKeywordsAndBothDrafts) {
    EXPECT_EQ(CheckError(R"({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "t", "description": "d", "$comment": "c", "default": {}, "examples": [{}],
        "type": "object",
        "properties": {
            "a": {"type": ["integer", "null"], "minimum": 0, "maximum": 9, "exclusiveMinimum": -1,
                  "exclusiveMaximum": 10, "multipleOf": 0.5},
            "b": {"type": "string", "minLength": 1, "maxLength": 8, "format": "date", "enum": ["x", "y"]},
            "c": {"type": "array", "items": {"const": 3}, "minItems": 0, "maxItems": 2, "uniqueItems": true},
            "d": {"allOf": [true], "anyOf": [{"type": "string"}], "oneOf": [false, {}], "not": {"type": "null"}}
        },
        "required": ["a"], "additionalProperties": false, "minProperties": 1, "maxProperties": 4
    })"), "");
    EXPECT_EQ(CheckError(R"({"$schema": "http://json-schema.org/draft-07/schema#"})"), "");
    EXPECT_EQ(CheckError("true"), "");
}

TEST(OrchJsonSchemaTest, RefusesKeywordsThatDifferBetweenDraftsOrArentSupported) {
    EXPECT_NE(CheckError(R"({"$ref": "#/$defs/x"})").find("references aren't supported"), std::string::npos);
    EXPECT_NE(CheckError(R"({"definitions": {}})").find("references aren't supported"), std::string::npos);
    EXPECT_NE(CheckError(R"({"pattern": "^a"})").find("regular expressions"), std::string::npos);
    EXPECT_NE(CheckError(R"({"dependencies": {}})").find("differs between"), std::string::npos);
    EXPECT_NE(CheckError(R"({"items": [{}, {}]})").find("prefixItems"), std::string::npos);
    EXPECT_NE(CheckError(R"({"exclusiveMinimum": true})").find("draft-4"), std::string::npos);
    EXPECT_NE(CheckError(R"({"properties": {"a/b": {"frobnicate": 1}}})").find("at /properties/a~1b/frobnicate"),
              std::string::npos);
    EXPECT_NE(CheckError(R"({"$schema": "http://json-schema.org/draft-04/schema#"})").find("draft-07 and 2020-12"),
              std::string::npos);
}

TEST(OrchJsonSchemaTest, RefusesInvalidKeywordValues) {
    for (const char* bad : {R"({"type": "float"})", R"({"type": []})", R"({"type": ["string", "string"]})",
                            R"({"enum": []})", R"({"required": ["a", "a"]})", R"({"required": [1]})",
                            R"({"minItems": -1})", R"({"maxLength": 1.5})", R"({"multipleOf": 0})",
                            R"({"anyOf": []})", R"({"allOf": [1]})", R"({"uniqueItems": 1})", R"({"properties": []})",
                            "3", R"({"not": "x"})"}) {
        EXPECT_NE(CheckError(bad), "") << bad;
    }
}

TEST(OrchJsonSchemaTest, RefusesSchemasNestedTooDeep) {
    std::string deep;
    for (int i = 0; i < 100; ++i) {
        deep += R"({"not": )";
    }
    deep += "{}" + std::string(100, '}');
    EXPECT_NE(CheckError(deep.c_str()).find("levels deep"), std::string::npos);
}

// ---- validating values ---------------------------------------------------------------------

TEST(OrchJsonSchemaTest, ChecksTypes) {
    EXPECT_TRUE(Valid(R"({"type": "integer"})", "3"));
    EXPECT_TRUE(Valid(R"({"type": "integer"})", "3.0"));  // draft-6 on: a zero fraction is an integer
    EXPECT_FALSE(Valid(R"({"type": "integer"})", "3.5"));
    EXPECT_TRUE(Valid(R"({"type": "number"})", "3"));
    EXPECT_FALSE(Valid(R"({"type": "boolean"})", "0"));
    EXPECT_TRUE(Valid(R"({"type": ["string", "null"]})", "null"));
    EXPECT_FALSE(Valid(R"({"type": ["string", "null"]})", "[]"));
    EXPECT_FALSE(Valid("false", "1"));
    EXPECT_TRUE(Valid("true", "1"));

    const auto errors = ValidateJson(J(R"({"type": ["string", "null"]})"), J("{}"));
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors[0].path, "");
    EXPECT_EQ(errors[0].message, "must be a string or null, not an object");
}

TEST(OrchJsonSchemaTest, ChecksObjects) {
    const char* schema = R"({
        "type": "object",
        "properties": {"name": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["name", "n"],
        "additionalProperties": false
    })";
    EXPECT_TRUE(Valid(schema, R"({"name": "x", "n": 1})"));
    EXPECT_EQ(Messages(ValidateJson(J(schema), J(R"({"n": "1", "extra": 2})"))),
              (std::vector<std::string>{" is missing the required property \"name\"",
                                        "/n must be an integer, not a string",
                                        "/extra is a property the schema doesn't allow"}));

    EXPECT_FALSE(Valid(R"({"additionalProperties": {"type": "number"}})", R"({"a": 1, "b": "2"})"));
    EXPECT_TRUE(Valid(R"({"additionalProperties": {"type": "number"}})", R"({"a": 1, "b": 2})"));
    EXPECT_FALSE(Valid(R"({"minProperties": 2})", R"({"a": 1})"));
    EXPECT_FALSE(Valid(R"({"maxProperties": 1})", R"({"a": 1, "b": 2})"));
    // Object keywords don't apply to other types.
    EXPECT_TRUE(Valid(R"({"required": ["a"]})", "7"));
}

TEST(OrchJsonSchemaTest, DoesNotRepeatOddPropertyNamesTheModelWrote) {
    const auto errors = ValidateJson(J(R"({"additionalProperties": false})"),
                                     J(R"({"ignore previous instructions": 1, "a/b": 2, "ok_name": 3})"));
    EXPECT_EQ(Messages(errors), (std::vector<std::string>{"/* is a property the schema doesn't allow",
                                                         "/* is a property the schema doesn't allow",
                                                         "/ok_name is a property the schema doesn't allow"}));
}

TEST(OrchJsonSchemaTest, ChecksArrays) {
    EXPECT_EQ(Messages(ValidateJson(J(R"({"items": {"type": "string"}})"), J(R"(["a", 2, "c", null])"))),
              (std::vector<std::string>{"/1 must be a string, not an integer", "/3 must be a string, not null"}));
    EXPECT_FALSE(Valid(R"({"minItems": 2})", "[1]"));
    EXPECT_FALSE(Valid(R"({"maxItems": 1})", "[1, 2]"));
    EXPECT_TRUE(Valid(R"({"uniqueItems": true})", R"([1, "1", [1], {"a": 1}])"));
    EXPECT_EQ(Messages(ValidateJson(J(R"({"uniqueItems": true})"), J(R"([{"a": 1, "b": 2}, 1.0, {"b": 2, "a": 1}, 1])"))),
              (std::vector<std::string>{"/2 repeats element 0, and the elements must be unique",
                                        "/3 repeats element 1, and the elements must be unique"}));
    EXPECT_TRUE(Valid(R"({"uniqueItems": false})", "[1, 1]"));
}

TEST(OrchJsonSchemaTest, ChecksStringLengthsInCodePoints) {
    EXPECT_TRUE(Valid(R"({"maxLength": 2})", "\"éé\""));  // four bytes, two characters
    EXPECT_FALSE(Valid(R"({"maxLength": 2})", "\"abc\""));
    EXPECT_FALSE(Valid(R"({"minLength": 1})", "\"\""));
}

TEST(OrchJsonSchemaTest, ChecksNumbers) {
    EXPECT_TRUE(Valid(R"({"minimum": 1, "maximum": 2})", "1"));
    EXPECT_TRUE(Valid(R"({"minimum": 1, "maximum": 2})", "2"));
    EXPECT_FALSE(Valid(R"({"exclusiveMinimum": 1})", "1"));
    EXPECT_FALSE(Valid(R"({"exclusiveMaximum": 2})", "2"));
    EXPECT_TRUE(Valid(R"({"multipleOf": 0.1})", "0.3"));
    EXPECT_FALSE(Valid(R"({"multipleOf": 0.1})", "0.35"));
    EXPECT_TRUE(Valid(R"({"multipleOf": 3})", "-9"));
    EXPECT_EQ(Messages(ValidateJson(J(R"({"minimum": 0.5})"), J("0.25"))),
              (std::vector<std::string>{" must be at least 0.5"}));
}

TEST(OrchJsonSchemaTest, ChecksEnumAndConstByJsonEquality) {
    EXPECT_TRUE(Valid(R"({"enum": ["a", 1, null, {"x": [1]}]})", "1.0"));
    EXPECT_TRUE(Valid(R"({"enum": ["a", 1, null, {"x": [1]}]})", R"({"x": [1.0]})"));
    EXPECT_FALSE(Valid(R"({"enum": ["a", 1, null]})", "\"A\""));
    EXPECT_FALSE(Valid(R"({"enum": [1]})", "true"));
    EXPECT_TRUE(Valid(R"({"const": {"a": 1, "b": 2}})", R"({"b": 2, "a": 1})"));
    EXPECT_FALSE(Valid(R"({"const": {"a": 1}})", R"({"a": 1, "b": 2})"));
    EXPECT_FALSE(Valid(R"({"const": false})", "0"));
}

TEST(OrchJsonSchemaTest, CombinesSchemas) {
    EXPECT_TRUE(Valid(R"({"allOf": [{"minimum": 1}, {"maximum": 3}]})", "2"));
    EXPECT_EQ(Messages(ValidateJson(J(R"({"allOf": [{"minimum": 1}, {"maximum": 3}]})"), J("5"))),
              (std::vector<std::string>{" must be at most 3"}));
    EXPECT_TRUE(Valid(R"({"anyOf": [{"type": "string"}, {"minimum": 3}]})", "4"));
    EXPECT_EQ(Messages(ValidateJson(J(R"({"anyOf": [{"type": "string"}, {"minimum": 3}]})"), J("2"))),
              (std::vector<std::string>{" must match at least one of the schema's anyOf alternatives"}));
    EXPECT_TRUE(Valid(R"({"oneOf": [{"type": "integer"}, {"type": "string"}]})", "1"));
    EXPECT_EQ(Messages(ValidateJson(J(R"({"oneOf": [{"type": "integer"}, {"type": "number"}]})"), J("1"))),
              (std::vector<std::string>{" must match exactly one of the schema's oneOf alternatives, not 2"}));
    EXPECT_FALSE(Valid(R"({"oneOf": [{"type": "integer"}, {"type": "string"}]})", "null"));
    EXPECT_FALSE(Valid(R"({"not": {"type": "null"}})", "null"));
    EXPECT_TRUE(Valid(R"({"not": {"type": "null"}})", "0"));
}

TEST(OrchJsonSchemaTest, NestedFailuresFailTheParent) {
    // A failure inside a property must count in an anyOf, which validates without recording.
    const char* schema = R"({"anyOf": [{"properties": {"a": {"type": "string"}}}, {"type": "array"}]})";
    EXPECT_FALSE(Valid(schema, R"({"a": 1})"));
    EXPECT_TRUE(Valid(schema, R"({"a": "x"})"));
    EXPECT_FALSE(Valid(R"({"not": {"properties": {"a": {"type": "string"}}}})", R"({"a": "x"})"));
    EXPECT_TRUE(Valid(R"({"not": {"properties": {"a": {"type": "string"}}}})", R"({"a": 1})"));
}

TEST(OrchJsonSchemaTest, EscapesPathsAndCapsTheErrors) {
    EXPECT_EQ(Messages(ValidateJson(J(R"({"properties": {"a/b~c": {"type": "string"}}})"), J(R"({"a/b~c": 1})"))),
              (std::vector<std::string>{"/a~1b~0c must be a string, not an integer"}));
    EXPECT_EQ(ValidateJson(J(R"({"items": {"type": "string"}})"), J("[1,2,3,4,5,6,7,8,9,10]"), 3).size(), 3u);
    EXPECT_EQ(ValidateJson(J(R"({"items": {"type": "string"}})"), J("[1,2,3,4,5,6,7,8,9,10]"), 0).size(), 1u);
}

TEST(OrchJsonSchemaTest, ValidateRefusesAnUncheckedSchema) {
    EXPECT_THROW((void)ValidateJson(J(R"({"$ref": "#"})"), J("1")), std::invalid_argument);
}

// ---- strict mode ---------------------------------------------------------------------------

TEST(OrchJsonSchemaTest, StrictModeNeedsClosedObjectsAndNoUnsupportedConstraints) {
    EXPECT_EQ(StrictModeProblem(J(R"({
        "type": "object",
        "properties": {
            "where": {"type": "string", "format": "date"},
            "tags": {"type": "array", "items": {"type": "object", "properties": {"k": {"enum": ["a", 1, null]}},
                                                 "additionalProperties": false}, "minItems": 1},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        },
        "required": ["where"],
        "additionalProperties": false
    })")), "");
    EXPECT_NE(StrictModeProblem(J(R"({"type": "object", "properties": {}})")).find("additionalProperties"),
              std::string::npos);
    EXPECT_NE(StrictModeProblem(J(R"({"type": "object", "additionalProperties": {"type": "string"}})")), "");
    EXPECT_EQ(StrictModeProblem(J(R"({
        "type": "object", "additionalProperties": false,
        "properties": {"inner": {"type": "object", "properties": {}}}
    })")), "/properties/inner describes an object without \"additionalProperties\": false, which strict mode needs");
    for (const char* key : {"minimum", "maximum", "multipleOf", "minLength", "maxLength", "maxItems"}) {
        const std::string schema = std::string(R"({"type": "object", "additionalProperties": false, "properties": {"x": {")") +
                                   key + R"(": 1}}})";
        EXPECT_NE(StrictModeProblem(J(schema.c_str())).find(key), std::string::npos) << key;
    }
    EXPECT_NE(StrictModeProblem(J(R"({"items": {}, "minItems": 2})")).find("minItems"), std::string::npos);
    EXPECT_NE(StrictModeProblem(J(R"({"oneOf": [{}]})")).find("oneOf"), std::string::npos);
    EXPECT_NE(StrictModeProblem(J(R"({"enum": [[1]]})")).find("enum"), std::string::npos);
}

// ---- the registry --------------------------------------------------------------------------

Tool AddTool() {
    Tool t;
    t.name = "add";
    t.description = "Adds two integers.";
    t.input_schema = J(R"({
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": false
    })");
    t.handler = [](const JsonValue& input) -> ToolOutcome {
        return {std::to_string(input.find("a")->as_int64() + input.find("b")->as_int64())};
    };
    return t;
}

std::string AddError(Tool tool) {
    try {
        ToolRegistry().add(std::move(tool));
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

TEST(OrchToolRegistryTest, MakesSpecsInTheOrderToolsWereAdded) {
    ToolRegistry registry;
    Tool strict = AddTool();
    strict.name = "add_strict";
    strict.strict = true;
    registry.add(AddTool()).add(std::move(strict));

    ASSERT_EQ(registry.size(), 2u);
    const std::vector<ToolSpec> specs = registry.specs();
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].name, "add");
    EXPECT_EQ(specs[0].description, "Adds two integers.");
    EXPECT_FALSE(specs[0].strict);
    EXPECT_EQ(WriteJsonCompact(specs[0].input_schema), WriteJsonCompact(AddTool().input_schema));
    EXPECT_EQ(specs[1].name, "add_strict");
    EXPECT_TRUE(specs[1].strict);
    EXPECT_NE(registry.find("add"), nullptr);
    EXPECT_EQ(registry.find("sub"), nullptr);
}

TEST(OrchToolRegistryTest, RefusesBadTools) {
    Tool t = AddTool();
    t.name = "has space";
    EXPECT_NE(AddError(t).find("tool name"), std::string::npos);

    t = AddTool();
    t.handler = nullptr;
    EXPECT_NE(AddError(t).find("no handler"), std::string::npos);

    t = AddTool();
    t.input_schema = J(R"({"type": "string"})");
    EXPECT_NE(AddError(t).find("\"type\": \"object\""), std::string::npos);

    t = AddTool();
    t.input_schema = J(R"({"type": "object", "properties": {"p": {"pattern": "a"}}})");
    EXPECT_NE(AddError(t).find("tool \"add\": JSON Schema at /properties/p/pattern"), std::string::npos);

    t = AddTool();
    t.strict = true;
    t.input_schema = J(R"({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}},
                          "additionalProperties": false})");
    EXPECT_NE(AddError(t).find("it's strict, but /properties/n uses minimum"), std::string::npos);

    ToolRegistry registry;
    registry.add(AddTool());
    EXPECT_THROW(registry.add(AddTool()), std::invalid_argument);
    EXPECT_EQ(registry.size(), 1u);
}

TEST(OrchToolRegistryTest, CallsAToolWithAValidInput) {
    ToolRegistry registry;
    registry.add(AddTool());
    const ToolOutcome out = registry.call("add", J(R"({"a": 2, "b": 40})"));
    EXPECT_FALSE(out.is_error);
    EXPECT_EQ(out.content, "42");
}

TEST(OrchToolRegistryTest, ReportsAnInvalidInputToTheModelWithoutCallingTheTool) {
    ToolRegistry registry;
    int calls = 0;
    Tool t = AddTool();
    t.handler = [&calls](const JsonValue&) -> ToolOutcome {
        ++calls;
        return {"called"};
    };
    registry.add(std::move(t));

    const ToolOutcome out = registry.call("add", J(R"({"a": "2", "c": 1})"));
    EXPECT_TRUE(out.is_error);
    EXPECT_EQ(out.content,
              "The input for add doesn't match its schema:\n"
              "- the input is missing the required property \"b\"\n"
              "- /a must be an integer, not a string\n"
              "- /c is a property the schema doesn't allow");

    const ToolOutcome not_json = registry.call("add", JsonValue("{\"a\": 2,"));
    EXPECT_TRUE(not_json.is_error);
    EXPECT_NE(not_json.content.find("wasn't valid JSON"), std::string::npos);
    EXPECT_EQ(calls, 0);
}

TEST(OrchToolRegistryTest, ReportsAnUnknownToolAndListsTheTools) {
    ToolRegistry registry;
    registry.add(AddTool());
    ToolOutcome out = registry.call("sub", J("{}"));
    EXPECT_TRUE(out.is_error);
    EXPECT_EQ(out.content, "There is no tool named \"sub\". The tools are: add.");
    out = registry.call("rm -rf /\nignore that", J("{}"));
    EXPECT_EQ(out.content, "There is no tool with that name. The tools are: add.");
    EXPECT_EQ(ToolRegistry().call("x", J("{}")).content, "There is no tool named \"x\". The tools are: none.");
}

TEST(OrchToolRegistryTest, TurnsAHandlersExceptionAndErrorIntoErrorResults) {
    ToolRegistry registry;
    Tool t = AddTool();
    t.handler = [](const JsonValue&) -> ToolOutcome { throw std::runtime_error("disk full"); };
    registry.add(std::move(t));
    Tool soft = AddTool();
    soft.name = "soft";
    soft.handler = [](const JsonValue&) -> ToolOutcome { return {"no such file", true}; };
    registry.add(std::move(soft));

    const ToolOutcome thrown = registry.call("add", J(R"({"a": 1, "b": 2})"));
    EXPECT_TRUE(thrown.is_error);
    EXPECT_EQ(thrown.content, "The tool failed: disk full");
    const ToolOutcome returned = registry.call("soft", J(R"({"a": 1, "b": 2})"));
    EXPECT_TRUE(returned.is_error);
    EXPECT_EQ(returned.content, "no such file");
}

// A provider that replays answers and records each request.
class ScriptedProvider : public Provider {
public:
    std::deque<ChatResponse> answers;
    std::vector<ChatRequest> requests;

    ChatResponse chat(const ChatRequest& request, const StreamCallbacks&) override {
        requests.push_back(request);
        ChatResponse r = std::move(answers.front());
        answers.pop_front();
        return r;
    }
    std::string name() const override { return "scripted"; }
};

ChatResponse ToolCalls(std::vector<ContentBlock> calls) {
    ChatResponse r;
    r.message.role = Message::Role::Assistant;
    r.message.content = std::move(calls);
    r.stop_reason = StopReason::ToolUse;
    return r;
}

TEST(OrchToolRegistryTest, DrivesTheToolLoop) {
    ToolRegistry registry;
    registry.add(AddTool());

    ScriptedProvider provider;
    provider.answers.push_back(ToolCalls({ContentBlock::ToolUse("c1", "add", J(R"({"a": 1, "b": 2})")),
                                          ContentBlock::ToolUse("c2", "add", J(R"({"a": 1})")),
                                          ContentBlock::ToolUse("c3", "mul", J(R"({"a": 1, "b": 2})"))}));
    ChatResponse done;
    done.message = Message::Assistant("3");
    provider.answers.push_back(done);

    ChatRequest request;
    request.model = "m";
    request.messages.push_back(Message::User("1 + 2?"));
    request.tools = registry.specs();
    const ToolLoopResult result = RunToolLoop(provider, request, registry.executor());

    EXPECT_EQ(result.turns, 2);
    EXPECT_EQ(result.last.message.text(), "3");
    ASSERT_EQ(provider.requests.size(), 2u);
    ASSERT_EQ(provider.requests[0].tools.size(), 1u);
    EXPECT_EQ(provider.requests[0].tools[0].name, "add");

    const Message& results = provider.requests[1].messages.back();
    ASSERT_EQ(results.content.size(), 3u);
    EXPECT_EQ(results.content[0].id, "c1");
    EXPECT_FALSE(results.content[0].is_error);
    EXPECT_EQ(results.content[0].text, "3");
    EXPECT_EQ(results.content[1].id, "c2");
    EXPECT_TRUE(results.content[1].is_error);
    EXPECT_NE(results.content[1].text.find("missing the required property \"b\""), std::string::npos);
    EXPECT_EQ(results.content[2].id, "c3");
    EXPECT_TRUE(results.content[2].is_error);
    EXPECT_NE(results.content[2].text.find("no tool named \"mul\""), std::string::npos);
}

}  // namespace
}  // namespace pulsatrix::orch
