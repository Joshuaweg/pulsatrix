# LLM Agents (Orchestrator)

pulsatrix is getting a native C++ library for LLM agents: a client for the Anthropic API and for
OpenAI-compatible local servers (llama-server, vLLM, Ollama), a tool loop, and later an MCP server
that lets Claude and other agents drive pulsatrix's explainers. It needs no Python and no SDK.

!!! note "In progress"
    Only the foundation is in so far (roadmap AGT-1): the build target, a holder for API keys and
    one-line JSON. The HTTP transport, the providers with the tool loop, and the tool registry
    follow as AGT-2 to AGT-4. See the [roadmap](roadmap/index.md#agt-agents-native-c).

## Building it

The orchestrator is a separate library, off by default, so the core library never takes on
networking dependencies:

```bash
cmake -S . -B build -DPULSATRIX_ENABLE_ORCHESTRATOR=ON
cmake --build build
```

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
