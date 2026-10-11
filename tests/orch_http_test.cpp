// AGT-2: the HTTP transport, against a local cpp-httplib server in this process. Every test runs
// on each backend the build has. With OpenSSL, the TLS tests make a throwaway certificate
// authority at run time, so no key is committed.
#include "pulsatrix/orch/http.hpp"

#include <gtest/gtest.h>
#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "pulsatrix/orch/sse.hpp"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#endif

namespace pulsatrix::orch {
namespace {

using namespace std::chrono_literals;

std::vector<HttpBackend> Backends() {
    std::vector<HttpBackend> out{HttpBackend::Httplib};
    if (HttpBackendAvailable(HttpBackend::Curl)) {
        out.push_back(HttpBackend::Curl);
    }
    return out;
}

const char* NameOf(HttpBackend b) { return b == HttpBackend::Curl ? "curl" : "httplib"; }

// Routes shared by the plain and TLS servers.
void AddRoutes(httplib::Server& server) {
    server.Post("/echo", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("X-Echo-Test", req.get_header_value("X-Test"));
        res.set_header("X-Echo-User-Agent", req.get_header_value("User-Agent"));
        res.set_content(req.body, "application/json");
    });
    server.Get("/teapot", [](const httplib::Request&, httplib::Response& res) {
        res.status = 418;
        res.set_content("{\"error\": \"teapot\"}", "application/json");
    });
    server.Get("/redirect", [](const httplib::Request&, httplib::Response& res) {
        res.set_redirect("/teapot");
    });
    server.Get("/big", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(1 << 20, 'x'), "text/plain");
    });
    server.Get("/slow", [](const httplib::Request&, httplib::Response& res) {
        std::this_thread::sleep_for(1500ms);
        res.set_content("late", "text/plain");
    });
    // Server-sent events in separate writes, so the client sees them arrive in pieces.
    server.Get("/events", [](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider("text/event-stream", [](std::size_t, httplib::DataSink& sink) {
            const char* parts[] = {"event: message_start\ndata: {\"n\": 1}\n\n", ": ping\n\n", "data: one\nda",
                                   "ta: two\n\n", "event: message_stop\ndata: {}\n\n"};
            for (const char* part : parts) {
                sink.write(part, std::strlen(part));
                std::this_thread::sleep_for(10ms);
            }
            sink.done();
            return true;
        });
    });
    // Never ends on its own.
    server.Get("/endless", [](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider("text/event-stream", [](std::size_t, httplib::DataSink& sink) {
            const std::string event = "data: " + std::string(1000, 'y') + "\n\n";
            for (int i = 0; i < 100000 && sink.is_writable(); ++i) {
                if (!sink.write(event.data(), event.size())) {
                    return false;
                }
            }
            sink.done();
            return true;
        });
    });
    server.Get("/overloaded", [](const httplib::Request&, httplib::Response& res) {
        res.status = 529;
        res.set_content("{\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\"}}", "application/json");
    });
}

// A server on a free port of 127.0.0.1, running until destroyed.
template <class Server>
class Running {
public:
    template <class... Args>
    explicit Running(Args&&... args) : server_(std::forward<Args>(args)...) {
#ifndef _WIN32
        std::signal(SIGPIPE, SIG_IGN);  // the server writes to clients that cancelled
#endif
        AddRoutes(server_);
        port_ = server_.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        server_.wait_until_ready();
    }
    ~Running() {
        server_.stop();
        thread_.join();
    }
    int port() const { return port_; }

private:
    Server server_;
    int port_ = 0;
    std::thread thread_;
};

class OrchHttpTest : public ::testing::Test {
protected:
    std::string url(const std::string& path) const { return "http://127.0.0.1:" + std::to_string(server_.port()) + path; }
    HttpRequest get(const std::string& path) const {
        HttpRequest r;
        r.method = "GET";
        r.url = url(path);
        return r;
    }
    Running<httplib::Server> server_;
};

TEST_F(OrchHttpTest, PostsAndReadsTheResponse) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        auto transport = MakeHttpTransport(backend);
        HttpRequest request;
        request.url = url("/echo");
        request.add_header("Content-Type", "application/json");
        request.add_header("X-Test", "hello");
        request.body = R"({"model": "x"})";
        HttpResponse response = transport->send(request);
        EXPECT_EQ(response.status, 200);
        EXPECT_TRUE(response.ok());
        EXPECT_EQ(response.body, R"({"model": "x"})");
        ASSERT_NE(response.header("x-echo-test"), nullptr);
        EXPECT_EQ(*response.header("X-ECHO-TEST"), "hello");  // any case
        EXPECT_EQ(*response.header("x-echo-user-agent"), "pulsatrix/" + std::string(PULSATRIX_TEST_VERSION));
        EXPECT_EQ(response.header("x-missing"), nullptr);
        for (const auto& header : response.headers) {
            EXPECT_EQ(header.first.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), std::string::npos) << header.first;
        }
    }
}

TEST_F(OrchHttpTest, AnErrorStatusIsAResponse) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        HttpResponse response = MakeHttpTransport(backend)->send(get("/teapot"));
        EXPECT_EQ(response.status, 418);
        EXPECT_FALSE(response.ok());
        EXPECT_EQ(response.body, "{\"error\": \"teapot\"}");
    }
}

TEST_F(OrchHttpTest, NeverFollowsRedirects) {
    // Following one would send the request's credentials somewhere the caller didn't choose.
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        HttpResponse response = MakeHttpTransport(backend)->send(get("/redirect"));
        EXPECT_EQ(response.status, 302);
        ASSERT_NE(response.header("location"), nullptr);
    }
}

TEST_F(OrchHttpTest, StreamsServerSentEventsAsTheyArrive) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        std::vector<SseEvent> events;
        SseParser parser([&](const SseEvent& e) { events.push_back(e); });
        int chunks = 0;
        HttpResponse response = MakeHttpTransport(backend)->stream(get("/events"), [&](std::string_view chunk) {
            ++chunks;
            parser.feed(chunk);
            return true;
        });
        parser.finish();
        EXPECT_EQ(response.status, 200);
        EXPECT_TRUE(response.body.empty());  // it went to the handler
        EXPECT_GT(chunks, 1);
        ASSERT_EQ(events.size(), 3u);
        EXPECT_EQ(events[0].type, "message_start");
        EXPECT_EQ(events[1].data, "one\ntwo");
        EXPECT_EQ(events[2].type, "message_stop");
    }
}

TEST_F(OrchHttpTest, AStreamedErrorIsCollectedForItsMessage) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        bool called = false;
        HttpResponse response = MakeHttpTransport(backend)->stream(get("/overloaded"), [&](std::string_view) {
            called = true;
            return true;
        });
        EXPECT_EQ(response.status, 529);
        EXPECT_FALSE(called);
        EXPECT_NE(response.body.find("overloaded_error"), std::string::npos);
    }
}

TEST_F(OrchHttpTest, TheHandlerCanCancelAStream) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        std::size_t received = 0;
        HttpResponse response = MakeHttpTransport(backend)->stream(get("/endless"), [&](std::string_view chunk) {
            received += chunk.size();
            return received < 50'000;
        });
        EXPECT_EQ(response.status, 200);
        EXPECT_GE(received, 50'000u);
        EXPECT_LT(received, 10'000'000u);
    }
}

TEST_F(OrchHttpTest, RefusesABodyLargerThanTheLimit) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        auto transport = MakeHttpTransport(backend);
        HttpRequest big = get("/big");
        big.max_response_bytes = 1000;
        try {
            (void)transport->send(big);
            ADD_FAILURE() << "accepted a body over the limit";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::TooLarge) << e.what();
        }
        HttpRequest endless = get("/endless");
        endless.max_response_bytes = 100'000;
        try {
            (void)transport->stream(endless, [](std::string_view) { return true; });
            ADD_FAILURE() << "streamed past the limit";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::TooLarge) << e.what();
        }
    }
}

TEST_F(OrchHttpTest, TimesOutAnIdleServer) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        HttpRequest slow = get("/slow");
        slow.read_timeout = 300ms;  // libcurl rounds this up to 1 s, still under the server's 1.5 s
        try {
            (void)MakeHttpTransport(backend)->send(slow);
            ADD_FAILURE() << "didn't time out";
        } catch (const HttpError& e) {
            EXPECT_TRUE(e.kind() == HttpError::Kind::Timeout || e.kind() == HttpError::Kind::Protocol) << e.what();
        }
    }
}

TEST(OrchHttpConnectTest, ReportsARefusedConnection) {
    int port = 0;
    {
        Running<httplib::Server> gone;
        port = gone.port();
    }
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        HttpRequest request;
        request.method = "GET";
        request.url = "http://127.0.0.1:" + std::to_string(port) + "/";
        try {
            (void)MakeHttpTransport(backend)->send(request);
            ADD_FAILURE() << "connected to a closed port";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::Connect) << e.what();
        }
    }
}

TEST(OrchHttpRequestTest, RefusesHeaderInjectionAndBadMethods) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        auto transport = MakeHttpTransport(backend);
        HttpRequest request;
        request.url = "http://127.0.0.1:9/";
        request.add_header("X-Test", "a\r\nX-Injected: b");
        try {
            (void)transport->send(request);
            ADD_FAILURE() << "sent a header with CRLF";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::InvalidRequest);
            EXPECT_EQ(std::string(e.what()).find("X-Injected"), std::string::npos);  // the value isn't echoed
        }
        request.headers.clear();
        request.add_header("Bad Name", "x");
        EXPECT_THROW((void)transport->send(request), HttpError);
        request.headers.clear();
        request.method = "GE T";
        EXPECT_THROW((void)transport->send(request), HttpError);
    }
}

TEST(OrchHttpRequestTest, DescribeRedactsSecrets) {
    HttpRequest request;
    request.url = "https://api.anthropic.com/v1/messages";
    request.add_secret_header("x-api-key", Secret("sk-ant-very-secret"));
    request.add_header("anthropic-version", "2023-06-01");
    const std::string text = request.describe();
    EXPECT_EQ(text.find("sk-ant"), std::string::npos);
    EXPECT_NE(text.find("x-api-key: <redacted>"), std::string::npos);
    EXPECT_NE(text.find("anthropic-version: 2023-06-01"), std::string::npos);
    EXPECT_EQ(request.headers[0].value, "sk-ant-very-secret");  // still sent
}

TEST(OrchHttpUrlTest, ParsesHttpAndHttpsUrls) {
    HttpUrl u = ParseHttpUrl("https://api.anthropic.com/v1/messages?beta=true");
    EXPECT_TRUE(u.https);
    EXPECT_EQ(u.host, "api.anthropic.com");
    EXPECT_EQ(u.port, 443);
    EXPECT_EQ(u.target, "/v1/messages?beta=true");

    u = ParseHttpUrl("HTTP://localhost:8080");
    EXPECT_FALSE(u.https);
    EXPECT_EQ(u.port, 8080);
    EXPECT_EQ(u.target, "/");

    u = ParseHttpUrl("http://[::1]:11434/api?x=1");
    EXPECT_EQ(u.host, "::1");
    EXPECT_EQ(u.port, 11434);

    u = ParseHttpUrl("http://example.com?q");
    EXPECT_EQ(u.target, "/?q");
}

TEST(OrchHttpUrlTest, RefusesWhatItCantSendSafely) {
    for (const char* bad : {"ftp://example.com/", "example.com/v1", "https://user:pw@example.com/",
                            "https://example.com/#frag", "https://:443/", "https://example.com:/",
                            "https://example.com:0/", "https://example.com:65536/", "https://example.com:8a/",
                            "https://exa mple.com/", "https://example.com/\n", "http://[::1/", "http://[::1]x/",
                            "http://::1/"}) {
        SCOPED_TRACE(bad);
        try {
            (void)ParseHttpUrl(bad);
            ADD_FAILURE() << "accepted";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::InvalidRequest);
        }
    }
}

TEST(OrchHttpBackendTest, NamesItsLibraries) {
    for (HttpBackend backend : Backends()) {
        const std::string name = MakeHttpTransport(backend)->name();
        EXPECT_NE(name.find(backend == HttpBackend::Curl ? "libcurl " : "cpp-httplib 0.57.1"), std::string::npos) << name;
    }
    if (!HttpBackendAvailable(HttpBackend::Curl)) {
        EXPECT_THROW((void)MakeHttpTransport(HttpBackend::Curl), std::invalid_argument);
    }
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT

// A throwaway CA and a server certificate it signed, written as PEM files.
class Certificates {
public:
    explicit Certificates(const std::string& server_san) {
        dir_ = std::filesystem::temp_directory_path() /
               ("pulsatrix_orch_tls_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(dir_);
        EVP_PKEY* ca_key = EVP_EC_gen("P-256");
        X509* ca = Make(ca_key, nullptr, ca_key, "pulsatrix test CA",
                        {{NID_basic_constraints, "critical,CA:TRUE"}, {NID_key_usage, "critical,keyCertSign,cRLSign"},
                         {NID_subject_key_identifier, "hash"}});
        EVP_PKEY* key = EVP_EC_gen("P-256");
        X509* cert = Make(key, ca, ca_key, "pulsatrix test server",
                          {{NID_basic_constraints, "critical,CA:FALSE"}, {NID_key_usage, "critical,digitalSignature"},
                           {NID_ext_key_usage, "serverAuth"}, {NID_subject_key_identifier, "hash"},
                           {NID_authority_key_identifier, "keyid:always"}, {NID_subject_alt_name, server_san.c_str()}});
        Write(ca_file(), [&](FILE* f) { PEM_write_X509(f, ca); });
        Write(cert_file(), [&](FILE* f) { PEM_write_X509(f, cert); });
        Write(key_file(), [&](FILE* f) { PEM_write_PrivateKey(f, key, nullptr, nullptr, 0, nullptr, nullptr); });
        X509_free(cert);
        X509_free(ca);
        EVP_PKEY_free(key);
        EVP_PKEY_free(ca_key);
    }
    ~Certificates() {
        std::error_code ignored;
        std::filesystem::remove_all(dir_, ignored);
    }
    std::string ca_file() const { return (dir_ / "ca.pem").string(); }
    std::string cert_file() const { return (dir_ / "server.pem").string(); }
    std::string key_file() const { return (dir_ / "server.key").string(); }

private:
    struct Extension {
        int nid;
        const char* value;
    };

    static X509* Make(EVP_PKEY* key, X509* issuer, EVP_PKEY* issuer_key, const char* cn, std::vector<Extension> extensions) {
        X509* x = X509_new();
        X509_set_version(x, 2);
        ASN1_INTEGER_set(X509_get_serialNumber(x), issuer == nullptr ? 1 : 2);
        X509_gmtime_adj(X509_getm_notBefore(x), -3600);
        X509_gmtime_adj(X509_getm_notAfter(x), 24 * 3600);
        X509_set_pubkey(x, key);
        X509_NAME_add_entry_by_txt(X509_get_subject_name(x), "CN", MBSTRING_ASC,
                                   reinterpret_cast<const unsigned char*>(cn), -1, -1, 0);
        X509_set_issuer_name(x, X509_get_subject_name(issuer == nullptr ? x : issuer));
        X509V3_CTX ctx;
        X509V3_set_ctx_nodb(&ctx);
        X509V3_set_ctx(&ctx, issuer == nullptr ? x : issuer, x, nullptr, nullptr, 0);
        for (const Extension& e : extensions) {
            X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &ctx, e.nid, e.value);
            EXPECT_NE(ext, nullptr) << e.value;
            X509_add_ext(x, ext, -1);
            X509_EXTENSION_free(ext);
        }
        X509_sign(x, issuer_key, EVP_sha256());
        return x;
    }

    static void Write(const std::string& path, const std::function<void(FILE*)>& write) {
        FILE* f = std::fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr) << path;
        write(f);
        std::fclose(f);
    }

    std::filesystem::path dir_;
};

class OrchHttpsTest : public ::testing::Test {
protected:
    OrchHttpsTest() : certs_("IP:127.0.0.1"), server_(certs_.cert_file().c_str(), certs_.key_file().c_str()) {}

    HttpRequest echo(const std::string& host = "127.0.0.1") const {
        HttpRequest r;
        r.url = "https://" + host + ":" + std::to_string(server_.port()) + "/echo";
        r.body = "over TLS";
        return r;
    }

    Certificates certs_;
    Running<httplib::SSLServer> server_;
};

TEST_F(OrchHttpsTest, RefusesAServerTheSystemDoesntTrust) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        try {
            (void)MakeHttpTransport(backend)->send(echo());
            ADD_FAILURE() << "trusted a certificate from an unknown CA";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::Tls) << e.what();
        }
    }
}

TEST_F(OrchHttpsTest, TrustsTheGivenCa) {
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        HttpResponse response = MakeHttpTransport(backend, {certs_.ca_file()})->send(echo());
        EXPECT_EQ(response.status, 200);
        EXPECT_EQ(response.body, "over TLS");
    }
}

TEST(OrchHttpsHostnameTest, RefusesACertificateForAnotherHost) {
    Certificates certs("DNS:somewhere.else.invalid");
    Running<httplib::SSLServer> server(certs.cert_file().c_str(), certs.key_file().c_str());
    for (HttpBackend backend : Backends()) {
        SCOPED_TRACE(NameOf(backend));
        HttpRequest request;
        request.url = "https://127.0.0.1:" + std::to_string(server.port()) + "/echo";
        try {
            (void)MakeHttpTransport(backend, {certs.ca_file()})->send(request);
            ADD_FAILURE() << "accepted a certificate for another host";
        } catch (const HttpError& e) {
            EXPECT_EQ(e.kind(), HttpError::Kind::Tls) << e.what();
        }
    }
}

#else

TEST(OrchHttpsTest, ABuildWithoutTlsRefusesHttps) {
    EXPECT_FALSE(HttpsSupported(HttpBackend::Httplib));
    HttpRequest request;
    request.url = "https://127.0.0.1:9/";
    try {
        (void)MakeHttpTransport(HttpBackend::Httplib)->send(request);
        ADD_FAILURE() << "sent an HTTPS request without TLS";
    } catch (const HttpError& e) {
        EXPECT_EQ(e.kind(), HttpError::Kind::Tls);
    }
}

#endif

}  // namespace
}  // namespace pulsatrix::orch
