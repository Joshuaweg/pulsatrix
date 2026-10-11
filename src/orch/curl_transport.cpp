// The libcurl backend (PULSATRIX_ORCHESTRATOR_CURL). libcurl brings its own TLS: Schannel on
// Windows, so certificates are checked against the Windows store without OpenSSL.
#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "http_internal.hpp"

namespace pulsatrix::orch::detail {

namespace {

void GlobalInit() {
    // curl_global_init isn't thread-safe, so it runs once, before any handle exists.
    static std::once_flag once;
    std::call_once(once, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw HttpError(HttpError::Kind::Other, "curl_global_init failed");
        }
    });
}

HttpError::Kind KindOf(CURLcode code) {
    switch (code) {
        case CURLE_COULDNT_RESOLVE_HOST:
        case CURLE_COULDNT_RESOLVE_PROXY:
        case CURLE_COULDNT_CONNECT:
            return HttpError::Kind::Connect;
        case CURLE_OPERATION_TIMEDOUT:
            return HttpError::Kind::Timeout;
        case CURLE_SSL_CONNECT_ERROR:
        case CURLE_PEER_FAILED_VERIFICATION:
        case CURLE_SSL_CACERT_BADFILE:
        case CURLE_SSL_CERTPROBLEM:
        case CURLE_SSL_CIPHER:
        case CURLE_SSL_ENGINE_NOTFOUND:
        case CURLE_SSL_ISSUER_ERROR:
        case CURLE_SSL_PINNEDPUBKEYNOTMATCH:
        case CURLE_SSL_INVALIDCERTSTATUS:
        case CURLE_SSL_CRL_BADFILE:
        case CURLE_SSL_SHUTDOWN_FAILED:
            return HttpError::Kind::Tls;
        case CURLE_FILESIZE_EXCEEDED:
            return HttpError::Kind::TooLarge;
        case CURLE_RECV_ERROR:
        case CURLE_SEND_ERROR:
        case CURLE_GOT_NOTHING:
        case CURLE_PARTIAL_FILE:
        case CURLE_WEIRD_SERVER_REPLY:
        case CURLE_HTTP2:
        case CURLE_HTTP2_STREAM:
            return HttpError::Kind::Protocol;
        default:
            return HttpError::Kind::Other;
    }
}

struct Transfer {
    CURL* handle = nullptr;
    HttpResponse* response = nullptr;
    BodySink* sink = nullptr;

    int status() const {
        long code = 0;
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &code);
        return static_cast<int>(code);
    }
};

std::size_t OnHeader(char* data, std::size_t size, std::size_t count, void* user) {
    auto* transfer = static_cast<Transfer*>(user);
    std::string line(data, size * count);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
    }
    if (line.rfind("HTTP/", 0) == 0) {
        transfer->response->headers.clear();  // a new status line: an interim response came first
    } else if (const std::size_t colon = line.find(':'); colon != std::string::npos) {
        std::string value = line.substr(colon + 1);
        const std::size_t first = value.find_first_not_of(" \t");
        value = first == std::string::npos ? "" : value.substr(first, value.find_last_not_of(" \t") - first + 1);
        transfer->response->headers.emplace_back(Lowercase(line.substr(0, colon)), std::move(value));
    }
    return size * count;
}

std::size_t OnBody(char* data, std::size_t size, std::size_t count, void* user) {
    auto* transfer = static_cast<Transfer*>(user);
    return transfer->sink->accept(transfer->status(), data, size * count) ? size * count : CURL_WRITEFUNC_ERROR;
}

class CurlTransport final : public HttpTransport {
public:
    explicit CurlTransport(HttpTransportOptions options) : options_(std::move(options)) { GlobalInit(); }

    HttpResponse send(const HttpRequest& request) override { return perform(request, nullptr); }

    HttpResponse stream(const HttpRequest& request, const HttpChunkHandler& on_chunk) override {
        return perform(request, &on_chunk);
    }

    std::string name() const override {
        const curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
        return std::string("libcurl ") + info->version + " (" + (info->ssl_version != nullptr ? info->ssl_version : "no TLS") + ")";
    }

private:
    HttpResponse perform(const HttpRequest& request, const HttpChunkHandler* on_chunk) {
        const HttpUrl url = ValidateRequest(request);
        if (url.https && !CurlHasTls()) {
            throw HttpError(HttpError::Kind::Tls, "this libcurl has no TLS, so it can't fetch " + request.url);
        }

        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> handle(curl_easy_init(), &curl_easy_cleanup);
        if (!handle) {
            throw HttpError(HttpError::Kind::Other, "curl_easy_init failed");
        }
        CURL* h = handle.get();

        curl_slist* list = nullptr;
        bool has_user_agent = false;
        for (const HttpHeader& header : request.headers) {
            list = curl_slist_append(list, (header.name + ": " + header.value).c_str());
            has_user_agent = has_user_agent || Lowercase(header.name) == "user-agent";
        }
        if (!has_user_agent) {
            list = curl_slist_append(list, ("User-Agent: " + UserAgent()).c_str());
        }
        list = curl_slist_append(list, "Expect:");  // no 100-continue round trip before the body
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(list, &curl_slist_free_all);

        HttpResponse response;
        BodySink sink(request, on_chunk);
        Transfer transfer{h, &response, &sink};

        // The low-speed limit is how libcurl times out an idle connection: under 1 byte per
        // second for read_timeout, rounded up to whole seconds.
        const long idle_seconds = static_cast<long>((request.read_timeout.count() + 999) / 1000);

        curl_easy_setopt(h, CURLOPT_URL, request.url.c_str());
        curl_easy_setopt(h, CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 2L);
        if (!options_.ca_cert_file.empty()) {
            curl_easy_setopt(h, CURLOPT_CAINFO, options_.ca_cert_file.c_str());
        }
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(request.connect_timeout.count()));
        curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
        curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, std::max(1L, idle_seconds));
        curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers.get());
        if (request.method == "GET") {
            curl_easy_setopt(h, CURLOPT_HTTPGET, 1L);
        } else if (request.method == "HEAD") {
            curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
        } else {
            curl_easy_setopt(h, CURLOPT_CUSTOMREQUEST, request.method.c_str());
        }
        if (!request.body.empty() || request.method == "POST" || request.method == "PUT" || request.method == "PATCH") {
            curl_easy_setopt(h, CURLOPT_POSTFIELDS, request.body.data());
            curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.body.size()));
        }
        curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &OnHeader);
        curl_easy_setopt(h, CURLOPT_HEADERDATA, &transfer);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &OnBody);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &transfer);

        char error_text[CURL_ERROR_SIZE] = {};
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, error_text);

        const CURLcode code = curl_easy_perform(h);
        if (sink.too_large()) {
            ThrowTooLarge(request);
        }
        if (code != CURLE_OK && !(code == CURLE_WRITE_ERROR && sink.cancelled())) {
            const std::string detail = error_text[0] != '\0' ? error_text : curl_easy_strerror(code);
            throw HttpError(KindOf(code), "HTTP request failed (" + detail + "): " + request.method + " " + request.url);
        }
        response.status = transfer.status();
        response.body = sink.take_body();
        return response;
    }

    HttpTransportOptions options_;
};

}  // namespace

std::unique_ptr<HttpTransport> MakeCurlTransport(const HttpTransportOptions& options) {
    return std::make_unique<CurlTransport>(options);
}

bool CurlHasTls() {
    const curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    return (info->features & CURL_VERSION_SSL) != 0;
}

}  // namespace pulsatrix::orch::detail
