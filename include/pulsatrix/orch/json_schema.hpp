/** @file json_schema.hpp
 *  @brief A small JSON Schema validator for tool inputs.
 *  @ingroup orchestrator
 *
 * It supports only the keywords that mean the same in JSON Schema draft-7 and 2020-12, so one
 * schema works with the Anthropic API, OpenAI-compatible servers and MCP clients, whichever
 * draft they assume:
 *
 * - `type` (a name or an array of names), `enum`, `const`
 * - objects: `properties`, `required`, `additionalProperties`, `minProperties`, `maxProperties`
 * - arrays: `items` (one schema for every element), `minItems`, `maxItems`, `uniqueItems`
 * - strings: `minLength`, `maxLength` (in Unicode code points)
 * - numbers: `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` (numbers),
 *   `multipleOf`
 * - combining: `allOf`, `anyOf`, `oneOf`, `not`
 * - annotations, which are accepted and not checked: `title`, `description`, `default`,
 *   `examples`, `format`, `$comment`, and `$schema` naming draft-7 or 2020-12
 *
 * Any other keyword is refused when the schema is checked, rather than silently not enforced.
 * That includes `$ref`, `$defs`/`definitions` and `dependencies`, whose spelling or meaning
 * changed between the drafts, `items` as an array (2020-12's `prefixItems`), and `pattern`,
 * since a regular expression engine that is safe on hostile input is more than this needs.
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"

namespace pulsatrix::orch {

/** @brief One way a value fails a schema. */
struct SchemaError {
    /** @brief Where, as a JSON Pointer into the value: `""` for the whole value, `/items/0/name`. */
    std::string path;
    std::string message;
};

/**
 * @brief Checks that @p schema uses only the supported keywords, each with a valid value.
 * @throws std::invalid_argument naming the first problem and where it is in the schema.
 */
void CheckSchema(const JsonValue& schema);

/**
 * @brief Checks @p value against @p schema.
 * @return The ways it fails, at most @p max_errors of them; empty if it matches. A failed
 *         `anyOf`, `oneOf` or `not` is one error, without the reasons inside it.
 * @note Error messages never quote the value, which came from a model and may be large.
 * @throws std::invalid_argument if @p schema doesn't pass CheckSchema().
 */
[[nodiscard]] std::vector<SchemaError> ValidateJson(const JsonValue& schema, const JsonValue& value,
                                                    std::size_t max_errors = 16);

/**
 * @brief Why @p schema can't be used with Anthropic's strict tool use, or "" if it can.
 * @note Strict mode needs `"additionalProperties": false` on every object, and doesn't support
 *       numeric, string-length or most array constraints, `oneOf`, `not`, or `enum` values that
 *       are arrays or objects. Its limits across a whole request (20 strict tools, 24 optional
 *       parameters, 16 union-typed parameters) aren't checked here.
 * @throws std::invalid_argument if @p schema doesn't pass CheckSchema().
 */
[[nodiscard]] std::string StrictModeProblem(const JsonValue& schema);

/** @brief JSON equality, as `enum`, `const` and `uniqueItems` use it: numbers by value (1 equals
 *         1.0), objects regardless of member order. */
[[nodiscard]] bool JsonEquals(const JsonValue& a, const JsonValue& b);

}  // namespace pulsatrix::orch
