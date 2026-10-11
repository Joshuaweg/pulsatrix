# LLM Agents (Orchestrator)

pulsatrix is getting a native C++ library for LLM agents: a client for the Anthropic API and for
OpenAI-compatible local servers (llama-server, vLLM, Ollama), a tool loop, and later an MCP server
that lets Claude and other agents drive pulsatrix's explainers. It needs no Python and no SDK.

!!! note "In progress"
    So far: the build target, a holder for API keys, one-line JSON (AGT-1), and HTTP with
    streaming (AGT-2). The providers with the tool loop, and the tool registry, follow as AGT-3
    and AGT-4. See the [roadmap](roadmap/index.md#agt-agents-native-c).

## Building it

The orchestrator is a separate library, off by default, so the core library never takes on
networking dependencies:

```bash
cmake -S . -B build -DPULSATRIX_ENABLE_ORCHESTRATOR=ON
cmake --build build
```

HTTP comes from [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT, one header, fetched
at a pinned version). Two settings control TLS:

| Option | Default | What it does |
|---|---|---|
| `PULSATRIX_ORCHESTRATOR_OPENSSL` | `AUTO` | HTTPS through OpenSSL 3: `AUTO` uses it when CMake finds it, `ON` requires it, `OFF` builds plain HTTP only. Certificates are checked against the system's trust store, including Windows' and macOS'. |
| `PULSATRIX_ORCHESTRATOR_CURL` | `OFF` | Also builds a libcurl backend. libcurl brings its own TLS, Schannel on Windows, so it's the way to get HTTPS on Windows without OpenSSL. |

Plain HTTP is enough for a local server such as llama-server or Ollama. The Anthropic API needs
HTTPS.

Link `pulsatrix_orchestrator` in the same build, or `pulsatrix::orchestrator` from an installed
pulsatrix (`find_package(pulsatrix)` sets `pulsatrix_HAS_ORCHESTRATOR`). Its headers are under
`pulsatrix/orch/` and its names in `pulsatrix::orch`, since `pulsatrix::Agent` already names the
reinforcement-learning policy.

## API keys

`orch::Secret` holds a credential. It has no `operator<<` and doesn't convert to a string, so a
key can't reach a log line or an error message by accident. The one way to read it is `reveal()`:

```cpp
#include "pulsatrix/orch/secret.hpp"

std::optional<pulsatrix::orch::Secret> key = pulsatrix::orch::Secret::FromEnv("ANTHROPIC_API_KEY");
if (!key) {
    std::cerr << "set ANTHROPIC_API_KEY\n";
    return 1;
}
std::cout << "using key " << key->redacted() << "\n";  // "using key <secret, 108 characters>"
// headers.emplace_back("x-api-key", key->reveal());   // the only place the value goes
```

The memory holding the key is overwritten when the `Secret` is destroyed or assigned over.

## HTTP and streaming

`MakeHttpTransport()` returns a transport: libcurl when the build has it, cpp-httplib otherwise.
`send()` waits for the whole response; `stream()` passes a successful response's body to a
callback as it arrives, which is how LLM APIs stream their answers. `SseParser` turns those chunks
into server-sent events:

```cpp
#include "pulsatrix/orch/http.hpp"
#include "pulsatrix/orch/sse.hpp"

using namespace pulsatrix::orch;

HttpRequest request;
request.url = "https://api.anthropic.com/v1/messages";
request.add_secret_header("x-api-key", *key);            // redacted by request.describe()
request.add_header("anthropic-version", "2023-06-01");
request.add_header("content-type", "application/json");
request.body = R"({"model": "claude-opus-5-5", "max_tokens": 256, "stream": true,
                   "messages": [{"role": "user", "content": "Hello"}]})";

SseParser parser([](const SseEvent& event) {
    if (event.type == "content_block_delta") {
        std::cout << event.data << "\n";  // a JSON object; AGT-3 will parse it for you
    }                                     // other types, such as ping, are skipped
});
std::unique_ptr<HttpTransport> http = MakeHttpTransport();
HttpResponse response = http->stream(request, [&](std::string_view chunk) {
    parser.feed(chunk);
    return true;  // false cancels the request
});
parser.finish();
if (!response.ok()) {
    std::cerr << "HTTP " << response.status << ": " << response.body << "\n";  // the error, collected
}
```

What the transport does for you:

- **No redirects.** A redirect would send your key to a host you didn't choose, so a 3xx is
  returned as a response instead.
- **Limits.** `connect_timeout` (10 s) and `read_timeout`, the longest the server may go without
  sending a byte (10 minutes, since a model can think for a while). `max_response_bytes` (64 MiB)
  stops a server that never stops sending.
- **Certificate checks can't be turned off.** To trust a company proxy or your own server,
  pass its CA certificate: `MakeHttpTransport({"/path/to/ca.pem"})`. It replaces the system's
  trust store.
- **Errors.** A response with a 4xx or 5xx status is a response. A request that got no
  response throws `HttpError`, whose `kind()` says why: `Connect`, `Tls`, `Timeout`, `TooLarge`,
  `Protocol` or `InvalidRequest`. A header value with a CR or LF is refused before anything is
  sent, so a value can't inject headers of its own.

The parser follows the WHATWG specification: any line ending, chunks split anywhere, comments,
`id` and `retry`. It passes every event type on; skip the ones you don't handle, since APIs add
new ones.

On Linux and macOS, a server that closes the connection while OpenSSL is writing can raise
`SIGPIPE`, which ends the process by default. A program that streams should ignore it:
`std::signal(SIGPIPE, SIG_IGN);`.

## JSON

The orchestrator uses pulsatrix's own JSON (`pulsatrix/json.hpp`), not a separate library. Its
parser is strict and refuses nesting deeper than 256 levels, which is what reading a remote
server's responses calls for. `WriteJsonCompact()` writes a value on one line with no whitespace,
for request bodies and for newline-delimited protocols such as MCP over stdio:

```cpp
pulsatrix::JsonValue message{pulsatrix::JsonValue::Object{}};
message.add("role", "user");
message.add("content", "Which input mattered most?\nAnswer briefly.");
std::string body = pulsatrix::WriteJsonCompact(message);
// {"role":"user","content":"Which input mattered most?\nAnswer briefly."}
```

If your own code uses `nlohmann::json`, include `pulsatrix/viz/mime_bundle.hpp` and
`nlohmann::json j = value;` converts a `JsonValue`. pulsatrix itself doesn't depend on nlohmann.
