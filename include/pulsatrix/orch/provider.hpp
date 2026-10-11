/** @file provider.hpp
 *  @brief LLM providers: the Anthropic API and OpenAI-compatible servers, behind one interface.
 *  @ingroup orchestrator
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/orch/http.hpp"
#include "pulsatrix/orch/message.hpp"
#include "pulsatrix/orch/secret.hpp"

namespace pulsatrix::orch {

/** @brief Called as a streamed answer arrives. Either may be empty. */
struct StreamCallbacks {
    std::function<void(std::string_view text)> on_text = {};
    std::function<void(std::string_view thinking)> on_thinking = {};
};

/** @brief The provider answered with an error, or failed after its retries. */
class ProviderError : public std::runtime_error {
public:
    /**
     * @param status the HTTP status, or 0 for an error with none (a connection failure, an error
     *        event in the middle of a stream)
     * @param type the provider's error type, for example `overloaded_error`
     */
    ProviderError(const std::string& message, int status, std::string type, std::string request_id)
        : std::runtime_error(message), status_(status), type_(std::move(type)), request_id_(std::move(request_id)) {}

    [[nodiscard]] int status() const { return status_; }
    [[nodiscard]] const std::string& type() const { return type_; }
    [[nodiscard]] const std::string& request_id() const { return request_id_; }

private:
    int status_;
    std::string type_;
    std::string request_id_;
};

/** @brief When and how long to retry. */
struct RetryPolicy {
    /** @brief Retries after the first attempt. 0 turns retrying off. */
    int max_retries = 2;
    /** @brief The first backoff. Each retry doubles it up to max_backoff, and the wait is a
     *         random fraction of that ("full jitter"), so many clients don't retry in step. */
    std::chrono::milliseconds initial_backoff{1000};
    std::chrono::milliseconds max_backoff{30'000};
    /** @brief The longest `retry-after` to honor. A longer one fails at once instead of waiting. */
    std::chrono::milliseconds max_retry_after{60'000};
    /** @brief Waits between attempts; replaceable so tests don't sleep. Empty: really sleep. */
    std::function<void(std::chrono::milliseconds)> sleep;
    /** @brief The seed for the jitter. */
    uint64_t seed = 0x5eed;
};

/** @brief A model behind an API. */
class Provider {
public:
    virtual ~Provider() = default;

    /**
     * @brief Sends @p request and returns the model's answer, streaming it to @p stream as it
     *        arrives.
     * @throws ProviderError if the provider answered with an error or the retries ran out.
     * @throws HttpError for a request that can't be sent at all (a bad URL, no TLS).
     */
    [[nodiscard]] virtual ChatResponse chat(const ChatRequest& request, const StreamCallbacks& stream = {}) = 0;

    /** @brief For example `anthropic (https://api.anthropic.com)`. */
    [[nodiscard]] virtual std::string name() const = 0;
};

/** @brief Settings for the Anthropic Messages API. */
struct AnthropicOptions {
    Secret api_key;
    std::string base_url = "https://api.anthropic.com";
    std::string api_version = "2023-06-01";
    /** @brief Beta features to enable (the `anthropic-beta` header), for example a dated feature name. */
    std::vector<std::string> betas;
    /** @brief Cache the tool definitions, the system prompt and the conversation so far, so each
     *         turn of a tool loop pays full price only for what's new. */
    bool prompt_caching = true;
    RetryPolicy retry;
    std::chrono::milliseconds read_timeout{600'000};

    /**
     * @brief Options with the key from `ANTHROPIC_API_KEY`.
     * @throws std::runtime_error if the variable isn't set.
     */
    [[nodiscard]] static AnthropicOptions FromEnv();
};

/** @brief Settings for an OpenAI-compatible chat-completions server (llama-server, vLLM, Ollama). */
struct OpenAiCompatibleOptions {
    /** @brief The URL before `/chat/completions`, for example llama-server's
     *         `http://127.0.0.1:8080/v1` or Ollama's `http://127.0.0.1:11434/v1`. */
    std::string base_url = "http://127.0.0.1:8080/v1";
    /** @brief Sent as `Authorization: Bearer`, if the server needs one. */
    std::optional<Secret> api_key;
    RetryPolicy retry;
    std::chrono::milliseconds read_timeout{600'000};
};

/** @brief The Anthropic provider. @p transport defaults to MakeHttpTransport(). */
[[nodiscard]] std::unique_ptr<Provider> MakeAnthropicProvider(AnthropicOptions options,
                                                              std::shared_ptr<HttpTransport> transport = nullptr);

/** @brief The OpenAI-compatible provider. @p transport defaults to MakeHttpTransport(). */
[[nodiscard]] std::unique_ptr<Provider> MakeOpenAiCompatibleProvider(OpenAiCompatibleOptions options,
                                                                     std::shared_ptr<HttpTransport> transport = nullptr);

}  // namespace pulsatrix::orch
