# LLM Agents (Orchestrator)

pulsatrix is getting a native C++ library for LLM agents: a client for the Anthropic API and for
OpenAI-compatible local servers (llama-server, vLLM, Ollama), a tool loop, and later an MCP server
that lets Claude and other agents drive pulsatrix's explainers. It needs no Python and no SDK.

!!! note "In progress"
    So far: the build target, API keys and JSON (AGT-1), HTTP with streaming (AGT-2), and the
    providers with the tool loop (AGT-3). A tool registry that validates the model's tool inputs
    follows as AGT-4, and an MCP server as AGT-5. See the
    [roadmap](roadmap/index.md#agt-agents-native-c).

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

## Talking to a model

A `Provider` sends a conversation to a model and streams the answer back. There are two:

- `MakeAnthropicProvider()` for the Claude API;
- `MakeOpenAiCompatibleProvider()` for a server with OpenAI's chat-completions API, such as
  llama-server, vLLM or Ollama running on your own machine.

Both take the same `ChatRequest` and return the same `ChatResponse`, so code written for one
works with the other:

```cpp
#include "pulsatrix/orch/provider.hpp"

using namespace pulsatrix::orch;

std::unique_ptr<Provider> claude = MakeAnthropicProvider(AnthropicOptions::FromEnv());  // ANTHROPIC_API_KEY

ChatRequest request;
request.model = "claude-opus-5-5";
request.system = "Answer in one sentence.";
request.messages.push_back(Message::User("What does layer-wise relevance propagation conserve?"));

ChatResponse answer = claude->chat(request, {[](std::string_view text) { std::cout << text; }});
std::cout << "\n" << answer.usage.output_tokens << " tokens\n";
```

The callback prints the answer as it arrives. `answer.message` holds the whole thing.

For a local model, only the provider changes:

```cpp
OpenAiCompatibleOptions local;
local.base_url = "http://127.0.0.1:11434/v1";  // Ollama; llama-server's default is http://127.0.0.1:8080/v1
std::unique_ptr<Provider> model = MakeOpenAiCompatibleProvider(local);
request.model = "qwen2.5:7b";
```

### Tools and the tool loop

Give the model tools with `ToolSpec` (a name, a description and a JSON Schema for the input), and
`RunToolLoop()` does the rest. It asks the model, runs each tool the model calls, sends the
results back, and repeats until the model answers:

```cpp
#include "pulsatrix/orch/tool_loop.hpp"

ToolSpec add;
add.name = "add";
add.description = "Adds two integers a and b and returns their sum.";
add.input_schema = pulsatrix::ParseJson(R"({"type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"], "additionalProperties": false})");
add.strict = true;  // Claude guarantees inputs that match the schema
request.tools = {add};

ToolLoopResult result = RunToolLoop(*claude, request, [](const ToolCall& call) -> ToolOutcome {
    if (call.name != "add") {
        return {"unknown tool " + call.name, /*is_error=*/true};
    }
    return {std::to_string(call.input.find("a")->as_int64() + call.input.find("b")->as_int64())};
});
std::cout << result.last.message.text() << "\n";
```

A tool that throws is reported to the model as an error, so the model can try again.
`max_turns` (20) bounds the loop. AGT-4 will add a registry that checks each input against its
schema before your function sees it.

### What the providers handle for you

- **Retries.** Overloads (529), server errors (500, 502, 503, 504), timeouts and broken
  connections are retried twice, with a random wait that doubles each time. A rate limit (429)
  waits as long as the server's `retry-after` says. A 429 without `retry-after` from the Claude
  API is the monthly spend cap, which retrying can't fix, so it fails at once. Nothing is
  retried once part of the answer has reached your callback, since a retry would repeat it.
- **Errors.** `ProviderError` carries the HTTP status, the API's error type (for example
  `overloaded_error`) and the request ID to quote to support.
- **Prompt caching** (Claude). The tool definitions, the system prompt and the conversation so
  far are cached, so each turn of a tool loop pays full price only for what's new.
  `answer.usage.cache_read_input_tokens` shows it working. Prompts under 512 tokens aren't
  cached.
- **Thinking.** Current Claude models think before answering. Their thinking blocks have to be
  sent back unchanged in a tool loop, and they are; `on_thinking` shows the thinking as it
  streams.
- **Tool choice.** Current Claude models (Opus 5.5, Sonnet 5.5, Fable 5.1) reject forcing a
  particular tool with HTTP 400, so the providers never do. Set `allow_tool_calls = false` to
  tell the model not to call any.
- **Local models.** A small model sometimes writes tool arguments that aren't valid JSON. The
  provider keeps them as a string in `call.input`, so your tool can report the mistake back to
  the model instead of the loop failing.

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

The providers are built on this layer. Use it directly for an API they don't cover.


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
        std::cout << event.data << "\n";  // a JSON object; the providers above parse these
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
