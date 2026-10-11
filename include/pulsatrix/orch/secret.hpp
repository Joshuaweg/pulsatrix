/** @file secret.hpp
 *  @brief A credential such as an API key, which can't be printed by accident.
 *  @ingroup orchestrator
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace pulsatrix::orch {

/**
 * @brief Holds a credential, such as an LLM provider's API key.
 * @note There is no `operator<<` and no conversion to `std::string`, so a Secret can't end up in
 *       a log line or an error message by accident: the only way to read it is reveal(), which is
 *       easy to search for. redacted() describes it without its contents. The memory holding the
 *       value is overwritten when the Secret is destroyed or assigned over.
 */
class Secret {
public:
    /** @brief An empty secret. */
    Secret() = default;
    explicit Secret(std::string value);

    /**
     * @brief Reads the environment variable @p name.
     * @return The value, or std::nullopt if the variable isn't set or is empty.
     */
    [[nodiscard]] static std::optional<Secret> FromEnv(const char* name);

    Secret(const Secret& other);
    Secret(Secret&& other) noexcept;
    Secret& operator=(const Secret& other);
    Secret& operator=(Secret&& other) noexcept;
    ~Secret();

    /** @brief The value, to put in a request header. Never log it. */
    [[nodiscard]] const std::string& reveal() const { return value_; }

    [[nodiscard]] bool empty() const { return value_.empty(); }
    [[nodiscard]] std::size_t size() const { return value_.size(); }

    /** @brief A description safe to print, for example `<secret, 108 characters>`. */
    [[nodiscard]] std::string redacted() const;

private:
    void wipe() noexcept;

    std::string value_;
};

}  // namespace pulsatrix::orch
