#include "pulsatrix/orch/secret.hpp"

#include <cstdlib>
#include <utility>

namespace pulsatrix::orch {

Secret::Secret(std::string value) : value_(std::move(value)) {}

std::optional<Secret> Secret::FromEnv(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return Secret(value);
}

Secret::Secret(const Secret& other) : value_(other.value_) {}

// Moving a short string copies its characters out of the small-string buffer, so the source is
// wiped rather than trusted to be empty.
Secret::Secret(Secret&& other) noexcept : value_(std::move(other.value_)) { other.wipe(); }

Secret& Secret::operator=(const Secret& other) {
    if (this != &other) {
        wipe();
        value_ = other.value_;
    }
    return *this;
}

Secret& Secret::operator=(Secret&& other) noexcept {
    if (this != &other) {
        wipe();
        value_ = std::move(other.value_);
        other.wipe();
    }
    return *this;
}

Secret::~Secret() { wipe(); }

std::string Secret::redacted() const {
    if (value_.empty()) {
        return "<empty secret>";
    }
    return "<secret, " + std::to_string(value_.size()) + " characters>";
}

void Secret::wipe() noexcept {
    // Through a volatile pointer, so the compiler can't drop the stores as dead before the
    // memory is freed. The whole capacity, since a value assigned over a longer one leaves the
    // longer one's tail in the buffer.
    value_.resize(value_.capacity());
    volatile char* p = value_.data();
    for (std::size_t i = 0; i < value_.size(); ++i) {
        p[i] = '\0';
    }
    value_.clear();
}

}  // namespace pulsatrix::orch
