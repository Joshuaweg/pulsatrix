#include "pulsatrix/orch/http.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "http_internal.hpp"

namespace pulsatrix::orch {

namespace {

bool EqualsIgnoringCase(std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
           });
}

[[noreturn]] void Invalid(const std::string& message) { throw HttpError(HttpError::Kind::InvalidRequest, message); }

// RFC 9110 token characters, for methods and header names.
bool IsToken(std::string_view s) {
    if (s.empty()) {
        return false;
    }
    return std::all_of(s.begin(), s.end(), [](char c) {
        const auto u = static_cast<unsigned char>(c);
        return std::isalnum(u) != 0 || std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
    });
}

}  // namespace

void HttpRequest::add_header(std::string name, std::string value) {
    headers.push_back({std::move(name), std::move(value), false});
}

void HttpRequest::add_secret_header(std::string name, const Secret& secret) {
    headers.push_back({std::move(name), secret.reveal(), true});
}

std::string HttpRequest::describe() const {
    std::string out = method + " " + url;
    for (const HttpHeader& h : headers) {
        out += "\n  " + h.name + ": " + (h.sensitive ? "<redacted>" : h.value);
    }
    return out;
}

const std::string* HttpResponse::header(std::string_view name) const {
    for (const auto& [key, value] : headers) {
        if (EqualsIgnoringCase(key, name)) {
            return &value;
        }
    }
    return nullptr;
}

HttpUrl ParseHttpUrl(std::string_view url) {
    for (char c : url) {
        const auto u = static_cast<unsigned char>(c);
        if (u <= 0x20 || u == 0x7F) {
            Invalid("URL contains whitespace or a control character");
        }
    }
    HttpUrl out;
    const std::size_t scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos) {
        Invalid("URL has no scheme: expected http:// or https://");
    }
    const std::string scheme = detail::Lowercase(std::string(url.substr(0, scheme_end)));
    if (scheme == "https") {
        out.https = true;
        out.port = 443;
    } else if (scheme == "http") {
        out.port = 80;
    } else {
        Invalid("URL scheme '" + scheme + "' isn't http or https");
    }
    std::string_view rest = url.substr(scheme_end + 3);
    if (rest.find('#') != std::string_view::npos) {
        Invalid("URL has a fragment, which is never sent to a server");
    }
    const std::size_t authority_end = rest.find_first_of("/?");
    const std::string_view authority = rest.substr(0, authority_end);
    out.target = authority_end == std::string_view::npos ? "/" : std::string(rest.substr(authority_end));
    if (out.target.front() == '?') {
        out.target.insert(out.target.begin(), '/');
    }
    if (authority.find('@') != std::string_view::npos) {
        Invalid("URL has user info (user@host); send credentials in a header instead");
    }
    std::string_view host = authority;
    std::string_view port;
    bool has_port = false;
    if (!authority.empty() && authority.front() == '[') {
        const std::size_t close = authority.find(']');
        if (close == std::string_view::npos) {
            Invalid("URL has an unclosed IPv6 address");
        }
        host = authority.substr(1, close - 1);
        const std::string_view after = authority.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') {
                Invalid("URL has text after its IPv6 address");
            }
            has_port = true;
            port = after.substr(1);
        }
    } else if (const std::size_t colon = authority.find(':'); colon != std::string_view::npos) {
        host = authority.substr(0, colon);
        has_port = true;
        port = authority.substr(colon + 1);
    }
    if (host.empty()) {
        Invalid("URL has no host");
    }
    out.host = std::string(host);
    if (has_port) {
        if (port.empty() || port.size() > 5 || port.find_first_not_of("0123456789") != std::string_view::npos) {
            Invalid("URL port '" + std::string(port) + "' isn't a number from 1 to 65535");
        }
        const int value = std::stoi(std::string(port));
        if (value < 1 || value > 65535) {
            Invalid("URL port '" + std::string(port) + "' isn't a number from 1 to 65535");
        }
        out.port = static_cast<uint16_t>(value);
    }
    return out;
}

namespace detail {

std::string Lowercase(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string UserAgent() { return std::string("pulsatrix/") + PULSATRIX_ORCH_VERSION; }

HttpUrl ValidateRequest(const HttpRequest& request) {
    if (!IsToken(request.method)) {
        Invalid("HTTP method '" + request.method + "' isn't a valid token");
    }
    for (const HttpHeader& h : request.headers) {
        if (!IsToken(h.name)) {
            Invalid("header name '" + h.name + "' isn't a valid token");
        }
        // The value isn't quoted: it may be a credential.
        if (h.value.find_first_of(std::string_view("\r\n\0", 3)) != std::string::npos) {
            Invalid("header '" + h.name + "' has a CR, LF or NUL in its value");
        }
    }
    if (request.connect_timeout.count() <= 0 || request.read_timeout.count() <= 0) {
        Invalid("timeouts must be positive");
    }
    return ParseHttpUrl(request.url);
}

bool BodySink::accept(int status, const char* data, std::size_t size) {
    received_ += size;
    if (received_ > request_.max_response_bytes) {
        too_large_ = true;
        return false;
    }
    if (on_chunk_ != nullptr && status >= 200 && status < 300) {
        if (!(*on_chunk_)(std::string_view(data, size))) {
            cancelled_ = true;
            return false;
        }
        return true;
    }
    body_.append(data, size);
    return true;
}

void ThrowTooLarge(const HttpRequest& request) {
    throw HttpError(HttpError::Kind::TooLarge, "response body passed max_response_bytes (" +
                                                   std::to_string(request.max_response_bytes) + ") from " + request.url);
}

}  // namespace detail

bool HttpBackendAvailable(HttpBackend backend) {
    switch (backend) {
        case HttpBackend::Httplib:
            return true;
        case HttpBackend::Curl:
#ifdef PULSATRIX_ORCH_WITH_CURL
            return true;
#else
            return false;
#endif
    }
    return false;
}

bool HttpsSupported(HttpBackend backend) {
    switch (backend) {
        case HttpBackend::Httplib:
            return detail::HttplibHasTls();
        case HttpBackend::Curl:
#ifdef PULSATRIX_ORCH_WITH_CURL
            return detail::CurlHasTls();
#else
            return false;
#endif
    }
    return false;
}

std::unique_ptr<HttpTransport> MakeHttpTransport(HttpBackend backend, HttpTransportOptions options) {
    switch (backend) {
        case HttpBackend::Httplib:
            return detail::MakeHttplibTransport(options);
        case HttpBackend::Curl:
#ifdef PULSATRIX_ORCH_WITH_CURL
            return detail::MakeCurlTransport(options);
#else
            throw std::invalid_argument("this pulsatrix build has no libcurl backend (PULSATRIX_ORCHESTRATOR_CURL)");
#endif
    }
    throw std::invalid_argument("unknown HttpBackend");
}

std::unique_ptr<HttpTransport> MakeHttpTransport(HttpTransportOptions options) {
    return MakeHttpTransport(HttpBackendAvailable(HttpBackend::Curl) ? HttpBackend::Curl : HttpBackend::Httplib,
                             std::move(options));
}

}  // namespace pulsatrix::orch
