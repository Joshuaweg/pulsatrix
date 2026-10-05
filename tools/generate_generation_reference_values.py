"""Offline, one-time reference-value generator for tests/generation_test.cpp (LLM-4): pulsatrix's
logit processing checked against Hugging Face Transformers' own logits processors and warpers.

Not run by CTest/CI. Run it in the image of tools/lrp_reference.Dockerfile (or any environment
with torch and transformers) and transcribe the printed C++ initializer lists:

    python3 tools/generate_generation_reference_values.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1, transformers 5.18.0.

Fixed logits over a 12-token vocabulary (det() as in the other generators, no ties) and the
sequence [3, 5, 5, 9] go through each processor alone and through all of them in the order
GenerationMixin._get_logits_processor builds them: repetition penalty, min new tokens,
temperature, top-k, top-p, min-p. Removed tokens are -inf; the C++ side prints them as such.
"""

import torch
from transformers.generation.logits_process import (LogitsProcessorList, MinNewTokensLengthLogitsProcessor,
                                                    MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

LOGITS = torch.tensor([[1.25, -0.5, 2.0, 0.75, -1.5, 0.25, 3.0, -0.25, 1.5, 0.5, -1.0, 2.5]])
TOKENS = torch.tensor([[3, 5, 5, 9]])


def cpp_float(v):
    if v == float("-inf"):
        return "-kInf"
    text = "%.9g" % (float(v) + 0.0)
    return (text if any(c in text for c in ".e") else text + ".0") + "f"


def emit(name, values):
    print("const std::vector<float> k%s = {" % name)
    print("    " + ", ".join(cpp_float(v) for v in values.reshape(-1).tolist()) + ",")
    print("};")


def run(name, processors):
    emit(name, LogitsProcessorList(processors)(TOKENS, LOGITS.clone()))


if __name__ == "__main__":
    run("RepetitionPenalty", [RepetitionPenaltyLogitsProcessor(1.3)])
    run("TemperatureTopK", [TemperatureLogitsWarper(0.7), TopKLogitsWarper(4)])
    run("TopP", [TopPLogitsWarper(0.8)])
    run("MinP", [MinPLogitsWarper(0.1)])
    # min_new_tokens 2 with 1 new token so far (prompt length 3): EOS (token 6) is banned.
    run("MinNewTokens", [MinNewTokensLengthLogitsProcessor(3, 2, 6)])
    run("Combined", [RepetitionPenaltyLogitsProcessor(1.2), TemperatureLogitsWarper(0.8), TopKLogitsWarper(6),
                     TopPLogitsWarper(0.9), MinPLogitsWarper(0.05)])
