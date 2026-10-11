/** @file http.hpp
 *  @brief HTTP for the orchestrator: requests, responses and a transport with two backends.
 *  @ingroup orchestrator
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/orch/secret.hpp"

namespace pulsatrix::orch {

/** @brief One request header. A sensitive one (a credential) is redacted by HttpRequest::describe(). */
struct HttpHeader {
    std::string name;
    std::string value;
    bool sensitive = false;
};

/** @brief An HTTP request. Redirects are never followed, so a credential only goes to @ref url. */
struct HttpRequest {
    std::string method = "POST";
    /** @brief An `http://` or `https://` URL. */
    std::string url;
    std::vector<HttpHeader> headers;
    std::string body;
    /** @brief The longest the connection (and TLS handshake) may take. */
    std::chrono::milliseconds connect_timeout{10'000};
    /** @brief The longest the server may go without sending a byte. A streamed response can last
     *         longer than this in total, as long as it keeps sending. */
    std::chrono::milliseconds read_timeout{600'000};
    /** @brief The most response body bytes to accept, against a server that never stops. */
    std::size_t max_response_bytes = std::size_t{64} << 20;

    /** @brief Adds a header. */
    void add_header(std::string name, std::string value);
    /** @brief Adds a header whose value is @p secret, marked sensitive. */
    void add_secret_header(std::string name, const Secret& secret);

    /** @brief The method, URL and headers, with sensitive values redacted, for logs and errors. */
    [[nodiscard]] std::string describe() const;
};

/** @brief An HTTP response. A status of 400 or 500 is a response, not an error. */
struct HttpResponse {
    int status = 0;
    /** @brief Response headers in the order received, names in lowercase. */
    std::vector<std::pair<std::string, std::string>> headers;
    /** @brief The body. Empty when HttpTransport::stream() delivered it to its callback. */
    std::string body;

    /** @brief The first header named @p name (any case), or nullptr if there is none. */
    [[nodiscard]] const std::string* header(std::string_view name) const;
    [[nodiscard]] bool ok() const { return status >= 200 && status < 300; }
};

/** @brief A request that got no complete response: the connection, TLS, a timeout or a limit failed. */
class HttpError : public std::runtime_error {
public:
    enum class Kind {
        InvalidRequest,  ///< a malformed URL or header, before anything is sent
        Connect,         ///< DNS or TCP failed
        Tls,             ///< the handshake or certificate verification failed, or the build has no TLS
        Timeout,         ///< connect_timeout or read_timeout passed
        TooLarge,        ///< the body passed max_response_bytes
        Protocol,        ///< the response wasn't valid HTTP, or the connection closed mid-response
        Other,
    };
    HttpError(Kind kind, const std::string& message) : std::runtime_error(message), kind_(kind) {}
    [[nodiscard]] Kind kind() const { return kind_; }

private:
    Kind kind_;
};

/** @brief Receives a streamed body chunk by chunk. Return false to cancel the request. */
using HttpChunkHandler = std::function<bool(std::string_view chunk)>;

/** @brief Sends HTTP requests. One instance may be used by one thread at a time. */
class HttpTransport {
public:
    virtual ~HttpTransport() = default;

    /**
     * @brief Sends @p request and waits for the whole response.
     * @throws HttpError if no complete response arrived.
     */
    [[nodiscard]] virtual HttpResponse send(const HttpRequest& request) = 0;

    /**
     * @brief Sends @p request and passes a successful (2xx) response's body to @p on_chunk as it
     *        arrives, for server-sent events. Any other status's body is collected into
     *        HttpResponse::body instead, for the error message.
     * @note If @p on_chunk returns false the request is cancelled, and stream() returns normally
     *       with what had arrived.
     * @throws HttpError if no complete response arrived.
     */
    [[nodiscard]] virtual HttpResponse stream(const HttpRequest& request, const HttpChunkHandler& on_chunk) = 0;

    /** @brief The backend and its TLS library, for example `cpp-httplib 0.57.1 (OpenSSL 3.5.5)`. */
    [[nodiscard]] virtual std::string name() const = 0;
};

/** @brief The HTTP libraries a transport can use. */
enum class HttpBackend {
    Httplib,  ///< cpp-httplib, always built; HTTPS when the build found OpenSSL
    Curl,     ///< libcurl, with PULSATRIX_ORCHESTRATOR_CURL; its own TLS (Schannel on Windows)
};

/** @brief How a transport trusts servers. Certificate verification can't be turned off. */
struct HttpTransportOptions {
    /** @brief A PEM file of the CA certificates to trust instead of the system's, for example a
     *         company proxy's or a local server's own. Empty: the system's trust store. */
    std::string ca_cert_file;
};

/** @brief Whether this build has @p backend. */
[[nodiscard]] bool HttpBackendAvailable(HttpBackend backend);
/** @brief Whether @p backend can make HTTPS requests in this build. */
[[nodiscard]] bool HttpsSupported(HttpBackend backend);

/**
 * @brief A transport using @p backend.
 * @throws std::invalid_argument if this build doesn't have @p backend.
 */
[[nodiscard]] std::unique_ptr<HttpTransport> MakeHttpTransport(HttpBackend backend, HttpTransportOptions options = {});

/** @brief A transport using libcurl if this build has it, which you asked for by turning it on,
 *         and cpp-httplib otherwise. */
[[nodiscard]] std::unique_ptr<HttpTransport> MakeHttpTransport(HttpTransportOptions options = {});

/** @brief An `http` or `https` URL split into its parts. */
struct HttpUrl {
    bool https = false;
    std::string host;  ///< without brackets for an IPv6 literal
    uint16_t port = 0;  ///< the scheme's default when the URL has none
    std::string target;  ///< the path and query, at least `/`
};

/**
 * @brief Parses an `http://` or `https://` URL.
 * @throws HttpError (InvalidRequest) for another scheme, user info (`user@host`), a fragment, an
 *         empty host, a bad port, or whitespace or control characters.
 */
[[nodiscard]] HttpUrl ParseHttpUrl(std::string_view url);

}  // namespace pulsatrix::orch
