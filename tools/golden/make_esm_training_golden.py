"""Writes golden values for PLM-7: one masked-LM training step of transformers' EsmForMaskedLM, and
three AdamW steps, to compare EncoderLM's loss, gradients and updates against
(tests/protein_training_test.cpp).

    python3 tools/golden/make_esm_training_golden.py tests/fixtures/hf_tiny/esm

writes esm_training_golden.safetensors into the model directory:
    ids, keep, targets   F32 [2, L]   a padded batch of two sequences, about 15% of residues picked
                                      (masked, swapped or kept), -100 elsewhere
    loss                 F32 [1]      mean cross-entropy over the picked tokens
    grad.<name>          F32 [...]    every trained parameter's gradient, by its transformers name
    step_losses          F32 [3]      the loss before each of three AdamW steps (lr 1e-3, betas 0.9
                                      and 0.98, eps 1e-8, weight decay 0.01 on every parameter)
    final.<name>         F32 [...]    every parameter after the three steps
Not run by CI. Needs torch, transformers and safetensors.
"""

import argparse
import os

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, EsmForMaskedLM

SEQUENCES = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MQIFVKTLTGKTITLEV"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = EsmForMaskedLM.from_pretrained(a.model_dir, attn_implementation="eager", dtype=torch.float32)
    model.train()  # dropout is 0 in this config; train() only matters for the record
    batch = tok(SEQUENCES, return_tensors="pt", padding=True)
    ids, keep = batch["input_ids"].clone(), batch["attention_mask"]
    g = torch.Generator().manual_seed(3)
    targets = torch.full_like(ids, -100)
    aa = torch.tensor([tok.convert_tokens_to_ids(c) for c in "ACDEFGHIKLMNPQRSTVWY"])
    for n in range(ids.shape[0]):
        for l in range(1, int(keep[n].sum()) - 1):
            if torch.rand(1, generator=g).item() < 0.15:
                targets[n, l] = ids[n, l]
                r = torch.rand(1, generator=g).item()
                if r < 0.8:
                    ids[n, l] = tok.mask_token_id
                elif r < 0.9:
                    ids[n, l] = aa[torch.randint(0, 20, (1,), generator=g)]
    assert (targets != -100).sum() > 0

    def loss_fn():
        logits = model(ids, attention_mask=keep).logits
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    out = {"ids": ids.float(), "keep": keep.float(), "targets": targets.float()}
    loss = loss_fn()
    loss.backward()
    out["loss"] = loss.detach().reshape(1)
    for name, param in model.named_parameters():
        if param.grad is not None:  # the contact head and unused position table get none
            out["grad." + name] = param.grad.detach().clone().contiguous()
    model.zero_grad()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = loss_fn()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    out["step_losses"] = torch.tensor(losses)
    for name, param in model.named_parameters():
        if "grad." + name in out:
            out["final." + name] = param.detach().clone().contiguous()
    save_file(out, os.path.join(a.model_dir, "esm_training_golden.safetensors"),
              metadata={"torch": torch.__version__, "transformers": transformers.__version__})
    print("loss", out["loss"].item(), "step losses", losses)


if __name__ == "__main__":
    main()
