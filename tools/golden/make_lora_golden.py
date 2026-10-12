"""Offline, one-time generator for tests/fixtures/hf_tiny/qwen2_lora/ (TRN-7): a LoRA adapter made by
PEFT on the tiny Qwen2 model of tools/generate_hf_tiny_models.py, saved with PEFT's save_pretrained,
and PEFT's own logits and gradients for it.

Not run by CTest/CI. Run it from the repository root in the image of tools/lrp_reference.Dockerfile,
which pins peft:

    python3 tools/golden/make_lora_golden.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1, transformers 5.18.0,
peft 0.21.0.

The adapter is PEFT's default LoRA (lora_A Kaiming-uniform, lora_dropout 0) with r = 4, lora_alpha = 8
and target_modules "all-linear", which on Qwen2 is every projection but lm_head. PEFT starts lora_B at
zero, so the adapter would change nothing; lora_B is redrawn (seed 1, std 0.05) to make it matter.

reference.safetensors holds, for the token ids [[3, 17, 42, 5, 60, 8]]:
    logits     the adapted model's logits, (1, 6, 64);
    upstream   a fixed random tensor shaped like the logits (seed 2);
    <key>      for each adapter tensor, under its adapter_model.safetensors key, the gradient of
               sum(logits * upstream) with respect to it, in PEFT's (out, r) / (r, in) layout.
"""

import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import Qwen2ForCausalLM

BASE = "tests/fixtures/hf_tiny/qwen2"
OUT = "tests/fixtures/hf_tiny/qwen2_lora"
TOKENS = [[3, 17, 42, 5, 60, 8]]


def main():
    model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="eager").eval()
    torch.manual_seed(0)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", lora_dropout=0.0))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=generator) * 0.05)
    peft_model.save_pretrained(OUT)

    logits = peft_model(torch.tensor(TOKENS)).logits
    upstream = torch.randn(logits.shape, generator=torch.Generator().manual_seed(2))
    (logits * upstream).sum().backward()
    reference = {"logits": logits.detach().contiguous(), "upstream": upstream.contiguous()}
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            # PEFT's parameter names carry the adapter name; its saved keys don't.
            reference[name.replace(".default", "")] = p.grad.detach().contiguous()
    save_file(reference, os.path.join(OUT, "reference.safetensors"))
    print("wrote", OUT, "with", len(reference) - 2, "gradients")


if __name__ == "__main__":
    main()
