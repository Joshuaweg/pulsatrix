// The cpp-httplib backend. The only file that includes httplib.h (23,000 lines), so no other
// translation unit pays for it.
#include <httplib.h>

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "http_internal.hpp"

namespace pulsatrix::orch::detail {

namespace {

HttpError::Kind KindOf(httplib::Error error) {
    switch (error) {
        case httplib::Error::Connection:
        case httplib::Error::BindIPAddress:
        case httplib::Error::ProxyConnection:
        case httplib::Error::UnsupportedAddressFamily:
            return HttpError::Kind::Connect;
        case httplib::Error::ConnectionTimeout:
        case httplib::Error::Timeout:
            return HttpError::Kind::Timeout;
        case httplib::Error::SSLConnection:
        case httplib::Error::SSLLoadingCerts:
        case httplib::Error::SSLServerVerification:
        case httplib::Error::SSLServerHostnameVerification:
            return HttpError::Kind::Tls;
        case httplib::Error::ExceedMaxPayloadSize:
            return HttpError::Kind::TooLarge;
        case httplib::Error::Read:
        case httplib::Error::Write:
        case httplib::Error::ConnectionClosed:
        case httplib::Error::HTTPParsing:
            return HttpError::Kind::Protocol;
        default:
            return HttpError::Kind::Other;
    }
}

template <class Client>
void Configure(Client& client, const HttpRequest& request) {
    client.set_connection_timeout(std::chrono::duration_cast<std::chrono::microseconds>(request.connect_timeout));
    client.set_read_timeout(std::chrono::duration_cast<std::chrono::microseconds>(request.read_timeout));
    client.set_write_timeout(std::chrono::duration_cast<std::chrono::microseconds>(request.read_timeout));
    client.set_follow_location(false);
    client.set_keep_alive(false);
}

class HttplibTransport final : public HttpTransport {
public:
    explicit HttplibTransport(HttpTransportOptions options) : options_(std::move(options)) {}

    HttpResponse send(const HttpRequest& request) override { return perform(request, nullptr); }

    HttpResponse stream(const HttpRequest& request, const HttpChunkHandler& on_chunk) override {
        return perform(request, &on_chunk);
    }

    std::string name() const override {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        return std::string("cpp-httplib ") + CPPHTTPLIB_VERSION + " (OpenSSL " + OpenSSL_version(OPENSSL_VERSION_STRING) + ")";
#else
        return std::string("cpp-httplib ") + CPPHTTPLIB_VERSION + " (no TLS)";
#endif
    }

private:
    HttpResponse perform(const HttpRequest& request, const HttpChunkHandler* on_chunk) {
        const HttpUrl url = ValidateRequest(request);

        httplib::Request req;
        req.method = request.method;
        req.path = url.target;
        bool has_user_agent = false;
        for (const HttpHeader& h : request.headers) {
            req.headers.emplace(h.name, h.value);
            has_user_agent = has_user_agent || Lowercase(h.name) == "user-agent";
        }
        if (!has_user_agent) {
            req.headers.emplace("User-Agent", UserAgent());
        }
        req.body = request.body;

        HttpResponse response;
        BodySink sink(request, on_chunk);
        req.response_handler = [&](const httplib::Response& r) {
            response.status = r.status;
            return true;
        };
        req.content_receiver = [&](const char* data, std::size_t size, std::size_t /*offset*/, std::size_t /*total*/) {
            return sink.accept(response.status, data, size);
        };

        httplib::Result result = url.https ? send_https(url, request, req) : send_http(url, request, req);
        if (sink.too_large() || (!result && result.error() == httplib::Error::ExceedMaxPayloadSize)) {
            ThrowTooLarge(request);
        }
        if (!result && !sink.cancelled()) {
            throw HttpError(KindOf(result.error()), "HTTP request failed (" + httplib::to_string(result.error()) + "): " +
                                                        request.method + " " + request.url);
        }
        if (result) {
            response.status = result->status;
            for (const auto& [key, value] : result->headers) {
                response.headers.emplace_back(Lowercase(key), value);
            }
        }
        response.body = sink.take_body();
        return response;
    }

    httplib::Result send_http(const HttpUrl& url, const HttpRequest& request, const httplib::Request& req) {
        httplib::Client client(url.host, url.port);
        Configure(client, request);
        return client.send(req);
    }

    httplib::Result send_https(const HttpUrl& url, const HttpRequest& request, const httplib::Request& req) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient client(url.host, url.port);
        Configure(client, request);
        client.enable_server_certificate_verification(true);
        client.enable_server_hostname_verification(true);
        if (!options_.ca_cert_file.empty()) {
            client.set_ca_cert_path(options_.ca_cert_file);  // instead of the system's
        }
        return client.send(req);
#else
        (void)url;
        (void)req;
        throw HttpError(HttpError::Kind::Tls, "this pulsatrix build's cpp-httplib has no TLS (CMake didn't find OpenSSL 3), so it can't fetch " +
                                                  request.url);
#endif
    }

    HttpTransportOptions options_;
};

}  // namespace

std::unique_ptr<HttpTransport> MakeHttplibTransport(const HttpTransportOptions& options) {
    return std::make_unique<HttplibTransport>(options);
}

bool HttplibHasTls() {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    return true;
#else
    return false;
#endif
}

}  // namespace pulsatrix::orch::detail
