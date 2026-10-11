# LLM Agents (Orchestrator)

pulsatrix has a native C++ library for LLM agents. It is a client for the Anthropic API (Claude)
and for OpenAI-compatible local servers (llama-server, vLLM, Ollama), with streaming and
retries, a tool loop, and a tool registry that checks the model's inputs. It needs no Python
and no SDK.

!!! note "Planned"
    An [MCP](https://modelcontextprotocol.io) (Model Context Protocol) server, which will let
    Claude and other agents call pulsatrix's explainers as tools, serving the same tool
    registry. See the [roadmap](roadmap/index.md#agt-agents-native-c).

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

Plain HTTP is enough for a local server such as llama-server or Ollama. The Claude API needs
HTTPS.

!!! warning "Without OpenSSL, `AUTO` builds anyway"
    If CMake doesn't find OpenSSL, `AUTO` only prints a status line, and every `https://`
    request (every Claude call) then fails at run time with an `HttpError` of kind `Tls`. Pass
    `-DPULSATRIX_ORCHESTRATOR_OPENSSL=ON` to make a missing OpenSSL a configure error instead.
    `HttpsSupported(HttpBackend::Httplib)` tells a program at run time.

To use it from your own CMake project, link `pulsatrix_orchestrator` in the same build, or
`pulsatrix::orchestrator` from an installed pulsatrix:

```cmake
find_package(pulsatrix 1.1 REQUIRED)
if(pulsatrix_HAS_ORCHESTRATOR)
    target_link_libraries(my_app PRIVATE pulsatrix::orchestrator)
endif()
```

The package finds OpenSSL and libcurl for you. Linking the static libraries by hand needs
`-lssl -lcrypto` too (and `-lcurl` with the libcurl backend).

Its headers are under `pulsatrix/orch/` and its names in `pulsatrix::orch`, since
`pulsatrix::Agent` already names the reinforcement-learning policy.

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
`max_tokens`, the longest answer allowed, is 4096 unless you set it.

### Local servers

For a local model, only the provider and the model name change:

```cpp
OpenAiCompatibleOptions local;
local.base_url = "http://127.0.0.1:11434/v1";  // Ollama; llama-server's default is http://127.0.0.1:8080/v1
std::unique_ptr<Provider> model = MakeOpenAiCompatibleProvider(local);
request.model = "qwen2.5:7b";  // a model the server has, for example one pulled with `ollama pull`
```

- A server that wants a token (vLLM started with `--api-key`, or a hosted endpoint) gets it
  from `local.api_key = Secret("...")`; it's sent as a bearer token.
- Tool calls need a model that was trained for them, and a server set up to parse them; check
  your server's documentation.
- A tool's `strict` flag is passed on to the server, which may ignore it. The tool registry
  below checks inputs either way.

### Tools and the tool loop

Define each tool once in a `ToolRegistry`: a name, a description, a
[JSON Schema](https://json-schema.org) for its input and the function that runs it. The registry
gives the provider the tool definitions and gives `RunToolLoop()` an executor. The loop asks the
model, runs each tool the model calls, sends the results back, and repeats until the model
answers:

```cpp
#include "pulsatrix/orch/tool_registry.hpp"

ToolRegistry tools;
tools.add({"add", "Adds two numbers a and b and returns their sum.",
           pulsatrix::ParseJson(R"({"type": "object",
               "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
               "required": ["a", "b"], "additionalProperties": false})"),
           [](const pulsatrix::JsonValue& input) -> ToolOutcome {
               const double sum = input.find("a")->as_double() + input.find("b")->as_double();
               return {pulsatrix::WriteJsonCompact(sum)};
           },
           /*strict=*/true});  // strict tool use: Claude only writes inputs that match the schema
request.tools = tools.specs();

ToolLoopResult result = RunToolLoop(*claude, request, tools.executor());
std::cout << result.last.message.text() << "\n";
```

The registry checks every input against its tool's schema before the function sees it, so the
function above can rely on `a` and `b` being there and being numbers. (A schema's `"integer"`
admits `1.0` and numbers too large for `int64_t`, as JSON Schema says it should, so a function
that calls `as_int64()` can still throw.) Every mistake goes back to the model as an error
result it can act on, never as an exception in your program:

- an input that doesn't match the schema lists each problem and where it is, as a
  [JSON Pointer](https://www.rfc-editor.org/rfc/rfc6901) (`/a` is member `a`, `/items/0` the
  first element): `The input for add doesn't match its schema: - /a must be a number, not a
  string`;
- arguments that weren't valid JSON, which small local models sometimes write;
- a tool name that doesn't exist: `There is no tool named "sub". The tools are: add.`;
- a function that throws: `The tool failed: ` and the exception's message. A function can also
  return `{"message", /*is_error=*/true}` itself.

The messages never quote the model's values back to it.

`executor()` refers to the registry, so keep the registry alive while the loop runs.
`RunToolLoop()` takes any function from `ToolCall` to `ToolOutcome`, so a registry is optional.

The loop's fourth argument, `ToolLoopOptions`, has:

- `max_turns` (20): the most requests the loop makes. If the model still wants tools then,
  `result.hit_turn_limit` is true and the conversation ends on tool results, so
  `result.last.message.text()` may be empty.
- `stream`: callbacks for the streamed text and thinking, as in `chat()`.
- `on_message`: called with each message the loop adds, the model's answers and the tool
  results, for logging or showing progress.

`result.messages` is the whole conversation, ready to continue with another request, and
`result.usage` adds up the tokens of every request.

#### Which schemas work

JSON Schema has several versions ("drafts"). Claude, OpenAI-compatible servers and MCP clients
don't all assume the same one, so a schema may use only the keywords that mean the same in
draft-7 and 2020-12, the two in use:

| For | Keywords |
|---|---|
| any value | `type` (a name or an array of names), `enum`, `const` |
| objects | `properties`, `required`, `additionalProperties`, `minProperties`, `maxProperties` |
| arrays | `items` (one schema for every element), `minItems`, `maxItems`, `uniqueItems` |
| strings | `minLength`, `maxLength` (counted in characters, not bytes) |
| numbers | `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf` |
| combining | `allOf`, `anyOf`, `oneOf`, `not` |
| notes, not checked | `title`, `description`, `default`, `examples`, `format`, `$comment`, `$schema` |

`add()` refuses any other keyword rather than not enforce it. That includes `$ref` and
`$defs`/`definitions` (spelled differently in the two drafts; write the schema out), `items` as
an array (2020-12 calls it `prefixItems`), and `pattern` (check the string in the tool).
`ValidateJson()` and `CheckSchema()` in `json_schema.hpp` work on their own as well.

**Strict tool use.** With `strict = true`, Claude's sampling is constrained so that it can only
write inputs that match the schema. Strict mode supports fewer schemas, and `add()` refuses a
strict tool whose schema it can't take, with the reason. A strict schema needs
`"additionalProperties": false` on every object, and can't have:

- `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` or `multipleOf`;
- `minLength` or `maxLength`;
- `maxItems`, `uniqueItems`, or `minItems` above 1;
- `minProperties` or `maxProperties`;
- `oneOf` or `not`;
- `enum` values that are arrays or objects, or `true`/`false` in place of a schema.

`StrictModeProblem()` says the same for any schema. The API also limits a whole request to 20
strict tools, 24 optional parameters and 16 parameters with a choice of types (`anyOf` or a
`type` array) across its strict tools. Those aren't checked here; the API answers 400.

### A complete program

```cpp
#include <csignal>
#include <iostream>
#include <memory>
#include <string>

#include "pulsatrix/orch/provider.hpp"
#include "pulsatrix/orch/tool_registry.hpp"

using namespace pulsatrix::orch;

int main(int argc, char** argv) {
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);  // see "HTTP and streaming" below
#endif
    const bool local = argc > 1 && std::string(argv[1]) == "--local";
    try {
        std::unique_ptr<Provider> model;
        ChatRequest request;
        if (local) {
            OpenAiCompatibleOptions options;
            options.base_url = "http://127.0.0.1:11434/v1";  // Ollama
            model = MakeOpenAiCompatibleProvider(options);
            request.model = "qwen2.5:7b";
        } else {
            model = MakeAnthropicProvider(AnthropicOptions::FromEnv());
            request.model = "claude-opus-5-5";
        }

        ToolRegistry tools;
        tools.add({"add", "Adds two numbers a and b and returns their sum.",
                   pulsatrix::ParseJson(R"({"type": "object",
                       "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"], "additionalProperties": false})"),
                   [](const pulsatrix::JsonValue& input) -> ToolOutcome {
                       return {pulsatrix::WriteJsonCompact(input.find("a")->as_double() +
                                                           input.find("b")->as_double())};
                   }});
        request.tools = tools.specs();
        request.messages.push_back(Message::User("What is 1234.5 + 6789.25? Use the add tool."));

        ToolLoopOptions options;
        options.stream.on_text = [](std::string_view text) { std::cout << text << std::flush; };
        const ToolLoopResult result = RunToolLoop(*model, request, tools.executor(), options);
        std::cout << "\n(" << result.turns << " requests, " << result.usage.input_tokens << " tokens in, "
                  << result.usage.output_tokens << " out)\n";
        return result.hit_turn_limit ? 2 : 0;
    } catch (const ProviderError& e) {
        std::cerr << "the provider failed: " << e.what() << "\n";
    } catch (const HttpError& e) {
        std::cerr << "the request couldn't be sent: " << e.what() << "\n";
    } catch (const std::exception& e) {  // a missing key, or a bad request
        std::cerr << e.what() << "\n";
    }
    return 1;
}
```

### What the providers handle for you

- **Retries.** These are retried, by default twice:
    - overloads (529) and server errors (500, 502, 503, 504);
    - request timeouts (408);
    - rate limits (429);
    - connections that fail, time out or break;
    - an overload or server error reported partway through Claude's stream.

    A retry waits a random time up to 1 s, then up to 2 s, doubling to at most 30 s, so many
    clients don't retry in step. When the server sends `retry-after`, the retry waits that long
    instead, but a `retry-after` over 60 s fails at once. A 429 without `retry-after` from the
    Claude API is the monthly spend cap, which retrying can't fix, so it fails at once too.
    Nothing is retried once part of the answer has reached your callback, since a retry would
    repeat it. `AnthropicOptions::retry` and `OpenAiCompatibleOptions::retry` (a `RetryPolicy`)
    change all of this; `max_retries = 0` turns it off.
- **Errors.** `chat()` and `RunToolLoop()` throw:
    - `ProviderError` for an error response, or when the retries run out. It carries the HTTP
      `status()`, the API's error `type()` (for example `overloaded_error`) and the
      `request_id()` to quote to support. A connection that kept failing has status 0 and type
      `transport`.
    - `HttpError` for a request that can't be sent at all: no TLS in this build (`Tls`), a bad
      URL or header (`InvalidRequest`), or a response over the size limit (`TooLarge`).
    - `std::invalid_argument` for a request without a model or messages, or with `max_tokens`
      below 1.

    Making the provider can fail too: `AnthropicOptions::FromEnv()` throws `std::runtime_error`
    when `ANTHROPIC_API_KEY` isn't set, and `MakeAnthropicProvider()` throws
    `std::invalid_argument` for an empty key.
- **Prompt caching** (Claude). The API can reuse the start of a prompt it saw recently, which
  makes it cheaper and faster. The providers mark the tool definitions, the system prompt and
  the conversation so far for caching, so each turn of a tool loop pays full price only for
  what's new. `answer.usage.cache_read_input_tokens` shows it working. Prompts under 512 tokens
  aren't cached.
- **Thinking.** Current Claude models reason before answering, and send that reasoning as
  "thinking" blocks. In a tool loop they have to be sent back unchanged, and they are;
  `on_thinking` shows the thinking as it streams.
- **Tool choice.** Current Claude models (Opus 5.5, Sonnet 5.5, Fable 5.1) reject forcing a
  particular tool with HTTP 400, so the providers never do. Set `allow_tool_calls = false` to
  tell the model not to call any.
- **Local models.** A small model sometimes writes tool arguments that aren't valid JSON. The
  OpenAI-compatible provider keeps them as a string in `call.input` instead of failing, and the
  registry reports the mistake back to the model.

### Threads

A provider handles one request at a time: give each thread its own. A `ToolRegistry` may run
tools from several threads at once, as long as your functions allow it, but finish adding tools
before any loop starts.

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
`MakeHttpTransport(HttpBackend::Httplib)` or `(HttpBackend::Curl)` picks one. `send()` waits for
the whole response; `stream()` passes a successful response's body to a callback as it arrives,
which is how LLM APIs stream their answers. `SseParser` turns those chunks into server-sent
events (SSE), the `text/event-stream` format the APIs stream in:

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
  trust store. A provider takes the transport as its second argument:
  `MakeAnthropicProvider(options, MakeHttpTransport({"/path/to/ca.pem"}))`.
- **Errors.** A response with a 4xx or 5xx status is a response. A request that got no
  response throws `HttpError`, whose `kind()` says why: `Connect`, `Tls`, `Timeout`, `TooLarge`,
  `Protocol`, `InvalidRequest` or `Other`. A header value with a CR or LF is refused before
  anything is sent, so a value can't inject headers of its own.
- **Threads.** A transport handles one request at a time.

The parser follows the SSE specification (part of the WHATWG HTML standard): any line ending,
chunks split anywhere, comments, `id` and `retry`. It passes every event type on; skip the ones
you don't handle, since APIs add new ones.

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
