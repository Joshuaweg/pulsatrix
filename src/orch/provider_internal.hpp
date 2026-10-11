// Shared by the providers (anthropic_provider.cpp, openai_provider.cpp): JSON helpers, error
// classification and the retry loop.
#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <string_view>

#include "pulsatrix/json.hpp"
#include "pulsatrix/orch/http.hpp"
#include "pulsatrix/orch/provider.hpp"

namespace pulsatrix::orch::detail {

JsonValue EmptyObject();
JsonValue EmptyArray();
/** @brief The string member @p key of @p object, or "" if it's missing or another type. */
std::string StringOr(const JsonValue& object, std::string_view key, std::string fallback = "");
/** @brief The integer member @p key of @p object, or @p fallback. */
int64_t IntOr(const JsonValue& object, std::string_view key, int64_t fallback = 0);
/** @brief @p object's member @p key if it's an object, or nullptr. */
const JsonValue* ObjectMember(const JsonValue& object, std::string_view key);
/** @brief A copy of @p object with member @p key set to @p value. */
JsonValue WithMember(const JsonValue& object, const std::string& key, JsonValue value);

/** @brief A failure the retry loop may retry: the error, plus the server's `retry-after`. */
struct RetryableFailure {
    ProviderError error;
    std::optional<std::chrono::milliseconds> retry_after;
};

/** @brief How a provider reports errors in a response body. */
enum class ErrorDialect { Anthropic, OpenAi };

/**
 * @brief Turns a non-2xx response into the exception to throw: RetryableFailure for 408, 429
 *        with `retry-after`, 500, 502, 503, 504 and 529, ProviderError otherwise. An Anthropic
 *        429 without `retry-after` is the monthly spend cap, which retrying can't fix.
 */
[[noreturn]] void ThrowForStatus(const HttpResponse& response, ErrorDialect dialect, const std::string& provider);

/**
 * @brief Runs @p attempt, retrying a RetryableFailure up to policy.max_retries times with
 *        exponential backoff and full jitter, or after the server's `retry-after`.
 * @note A transport failure (connect, timeout, a broken stream) is retryable too. Neither is
 *       retried once @p attempt has streamed part of the answer to the caller (it sets the flag
 *       it's given), since a retry would repeat that part.
 */
ChatResponse WithRetries(const RetryPolicy& policy, std::mt19937_64& rng, const std::string& provider,
                         const std::function<ChatResponse(bool& streamed)>& attempt);

/** @brief Wraps a chunk handler so an exception it throws is stored in @p failure and cancels the
 *         transfer, instead of unwinding through the HTTP library (libcurl is C). */
HttpChunkHandler Guarded(std::exception_ptr& failure, std::function<void(std::string_view)> handler);

/** @brief Validates the request fields every provider needs. */
void ValidateChatRequest(const ChatRequest& request);

}  // namespace pulsatrix::orch::detail
