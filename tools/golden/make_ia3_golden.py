"""Offline, one-time generator for tests/fixtures/hf_tiny/qwen2_ia3/ (TRN-10): an IA3 adapter made by PEFT
on the tiny Qwen2 model of tools/generate_hf_tiny_models.py, saved with PEFT's save_pretrained, and
PEFT's own logits and gradients for it.

Not run by CTest/CI. Run it from the repository root in the image of tools/lrp_reference.Dockerfile:

    python3 tools/golden/make_ia3_golden.py

Versions: Python 3.12, torch 2.14.1, transformers 5.18.0, peft 0.21.0.

The adapter rescales keys, values and the MLP's inner activations, as the IA3 paper does:
target_modules k_proj, v_proj and down_proj, with down_proj a feedforward module (its input is
scaled; the others' outputs are). k_proj and v_proj have biases in Qwen2, which IA3 scales too. PEFT
starts every vector at 1, which changes nothing; they are redrawn as 1 + 0.3 N(0, 1) (seed 1).

reference.safetensors holds, for the token ids [[3, 17, 42, 5, 60, 8]]:
    logits     the adapted model's logits, (1, 6, 64);
    upstream   a fixed random tensor shaped like the logits (seed 2);
    <key>      for each adapter vector, under its adapter_model.safetensors key, the gradient of
               sum(logits * upstream) with respect to it.
"""

import os

import torch
from peft import IA3Config, get_peft_model
from safetensors.torch import save_file
from transformers import Qwen2ForCausalLM

BASE = "tests/fixtures/hf_tiny/qwen2"
OUT = "tests/fixtures/hf_tiny/qwen2_ia3"
TOKENS = [[3, 17, 42, 5, 60, 8]]


def main():
    model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="eager").eval()
    config = IA3Config(target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"])
    peft_model = get_peft_model(model, config)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "ia3_l" in name:
                p.copy_(1 + 0.3 * torch.randn(p.shape, generator=generator))
    peft_model.save_pretrained(OUT)

    logits = peft_model(torch.tensor(TOKENS)).logits
    upstream = torch.randn(logits.shape, generator=torch.Generator().manual_seed(2))
    (logits * upstream).sum().backward()
    reference = {"logits": logits.detach().contiguous(), "upstream": upstream.contiguous()}
    for name, p in peft_model.named_parameters():
        if "ia3_l" in name:
            reference[name.replace(".default", "")] = p.grad.detach().contiguous()
    save_file(reference, os.path.join(OUT, "reference.safetensors"))
    print("wrote", OUT, "with", len(reference) - 2, "gradients:", sorted(reference)[:3])


if __name__ == "__main__":
    main()
