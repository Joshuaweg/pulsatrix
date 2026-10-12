"""Offline, one-time generator for tests/fixtures/hf_tiny/qwen2_loreft/ (TRN-9): LoReFT interventions
made by pyreft on the tiny Qwen2 model of tools/generate_hf_tiny_models.py, and the logits, gradients
and greedy generation they give.

Not run by CTest/CI. pyreft 0.1.0 pins transformers 4.45.1, so it is installed without its
dependencies next to the pinned stack of tools/lrp_reference.Dockerfile, and only its
LoreftIntervention class is used; a forward hook applies it to a decoder layer's output, as pyreft's
"block_output" component does:

    pip install --no-deps pyreft==0.1.0 pyvene==0.1.8 && pip install datasets==4.0.0
    python3 tools/golden/make_loreft_golden.py

Versions: Python 3.12, torch 2.14.1, transformers 5.18.0, pyreft 0.1.0, pyvene 0.1.8.

Interventions of rank 2 sit on the outputs of layers 0 and 1, at the first position and the last two
("f1+l2") of the prompt [[3, 17, 42, 5, 60, 8]]. pyreft starts learned_source as an nn.Linear and
rotate_layer orthonormal (torch's orthogonal parametrization); the parametrization is then removed,
keeping its value, so the gradient is taken with respect to R itself. learned_source is then redrawn
(seed 3, weight std 1, bias std 2) so that the edits change what the model generates.

loreft.safetensors holds, per layer i, pyreft's tensors in its own layouts:
    layer.<i>.rotate_layer           R, (hidden, rank), orthonormal columns;
    layer.<i>.learned_source.weight  W, (rank, hidden);
    layer.<i>.learned_source.bias    b, (rank,);
and metadata first = 1, last = 2.
reference.safetensors holds:
    logits      the intervened model's logits, (1, 6, 64);
    upstream    a fixed random tensor shaped like the logits (seed 2);
    grad.<key>  for each key above, the gradient of sum(logits * upstream);
    generated   8 greedy tokens after the prompt, intervening on the prompt's positions only.
"""

import os

import torch
import torch.nn.utils.parametrize as parametrize
from pyreft.interventions import LoreftIntervention
from safetensors.torch import save_file
from transformers import Qwen2ForCausalLM

BASE = "tests/fixtures/hf_tiny/qwen2"
OUT = "tests/fixtures/hf_tiny/qwen2_loreft"
TOKENS = [[3, 17, 42, 5, 60, 8]]
LAYERS = [0, 1]
FIRST, LAST = 1, 2


def positions(length):
    return sorted(set(range(min(FIRST, length))) | set(range(max(length - LAST, 0), length)))


def main():
    model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="eager").eval()
    torch.manual_seed(0)
    interventions = {}
    for i in LAYERS:
        intervention = LoreftIntervention(embed_dim=32, low_rank_dimension=2, dtype=torch.float32)
        parametrize.remove_parametrizations(intervention.rotate_layer, "weight", leave_parametrized=True)
        interventions[i] = intervention
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for intervention in interventions.values():
            intervention.learned_source.weight.copy_(torch.randn(intervention.learned_source.weight.shape, generator=generator))
            intervention.learned_source.bias.copy_(2 * torch.randn(intervention.learned_source.bias.shape, generator=generator))

    with torch.no_grad():
        base = model.generate(torch.tensor(TOKENS), max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0)
    state = {"prompt_length": None}

    def hook(i):
        def apply(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            length = hidden.shape[1]
            if state["prompt_length"] is not None and length != state["prompt_length"]:
                return output  # a generated token: pyreft intervenes on the prompt only
            at = positions(length)
            edited = hidden.clone()
            edited[:, at] = interventions[i](hidden[:, at])
            return (edited,) + tuple(output[1:]) if isinstance(output, tuple) else edited
        return apply

    for i in LAYERS:
        model.model.layers[i].register_forward_hook(hook(i))

    logits = model(torch.tensor(TOKENS)).logits
    upstream = torch.randn(logits.shape, generator=torch.Generator().manual_seed(2))
    (logits * upstream).sum().backward()

    weights, reference = {}, {"logits": logits.detach().contiguous(), "upstream": upstream.contiguous()}
    for i, intervention in interventions.items():
        for name, p in [("rotate_layer", intervention.rotate_layer.weight),
                        ("learned_source.weight", intervention.learned_source.weight),
                        ("learned_source.bias", intervention.learned_source.bias)]:
            key = "layer.%d.%s" % (i, name)
            weights[key] = p.detach().contiguous()
            reference["grad." + key] = p.grad.detach().contiguous()

    state["prompt_length"] = len(TOKENS[0])
    with torch.no_grad():
        generated = model.generate(torch.tensor(TOKENS), max_new_tokens=8, do_sample=False, eos_token_id=None,
                                   pad_token_id=0)
    reference["generated"] = generated[0, len(TOKENS[0]):].to(torch.float32).contiguous()

    os.makedirs(OUT, exist_ok=True)
    save_file(weights, os.path.join(OUT, "loreft.safetensors"), metadata={"first": str(FIRST), "last": str(LAST)})
    save_file(reference, os.path.join(OUT, "reference.safetensors"))
    print("wrote", OUT, "generated", generated[0, len(TOKENS[0]):].tolist(), "base", base[0, len(TOKENS[0]):].tolist())


if __name__ == "__main__":
    main()
