// Shared by the HTTP backends (http.cpp, httplib_transport.cpp, curl_transport.cpp).
#pragma once

#include <memory>
#include <string>

#include "pulsatrix/orch/http.hpp"

namespace pulsatrix::orch::detail {

/** @brief Checks the method and headers and parses the URL, so both backends refuse the same
 *         requests. A CR or LF in a header would let a value inject headers of its own. */
HttpUrl ValidateRequest(const HttpRequest& request);

/** @brief `pulsatrix/<version>`, sent unless the request sets its own User-Agent. */
std::string UserAgent();

/** @brief Lowercase ASCII. */
std::string Lowercase(std::string s);

/** @brief Collects a response body up to a limit, or passes a 2xx body to a handler. */
class BodySink {
public:
    BodySink(const HttpRequest& request, const HttpChunkHandler* on_chunk) : request_(request), on_chunk_(on_chunk) {}

    /** @brief Takes one chunk once the status is known. False stops the transfer; see
     *         cancelled() and too_large() for why. */
    bool accept(int status, const char* data, std::size_t size);

    [[nodiscard]] bool cancelled() const { return cancelled_; }
    [[nodiscard]] bool too_large() const { return too_large_; }
    std::string take_body() { return std::move(body_); }

private:
    const HttpRequest& request_;
    const HttpChunkHandler* on_chunk_;
    std::string body_;
    std::size_t received_ = 0;
    bool cancelled_ = false;
    bool too_large_ = false;
};

[[noreturn]] void ThrowTooLarge(const HttpRequest& request);

std::unique_ptr<HttpTransport> MakeHttplibTransport(const HttpTransportOptions& options);
bool HttplibHasTls();
#ifdef PULSATRIX_ORCH_WITH_CURL
std::unique_ptr<HttpTransport> MakeCurlTransport(const HttpTransportOptions& options);
bool CurlHasTls();
#endif

}  // namespace pulsatrix::orch::detail
