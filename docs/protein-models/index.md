# Protein Language Models

A protein language model reads an amino-acid sequence the way a text model reads words. ESM-2,
the model family pulsatrix runs, was trained by hiding residues and predicting them. Along the
way it learned which positions tolerate change, which residues touch in the folded protein, and
which motifs mark functional sites. pulsatrix loads the published ESM-2 checkpoints natively in
C++, on CPU or GPU, with no Python. You can run them, read their internal representations,
train them further and explain their predictions.

This is the start of the PLM epic. Variant scoring, contact maps, protein views and explanations
come next (see the [roadmap](../roadmap/index.md#plm-protein-language-models) and the
[research and plan](../roadmap/protein-language-models.md)).

## What's inside

- **`EncoderLM`** (`encoder_lm.hpp`): an ESM-2 masked language model. It has:
  - token embeddings with ESM's token-dropout scaling;
  - pre-LayerNorm encoder layers with rotary attention (`EncoderBlock`);
  - a final LayerNorm;
  - the LM head, which maps each position to scores for all 33 tokens.
- **`LoadEncoderLM(directory, backend)`**: loads a Hugging Face ESM-2 directory (`config.json`
  plus safetensors), such as
  [`facebook/esm2_t6_8M_UR50D`](https://huggingface.co/facebook/esm2_t6_8M_UR50D) up to
  `esm2_t33_650M_UR50D`. ESM-2 is MIT-licensed.
- **`LoadEsmTokenizer(vocab.txt)`**: ESM's tokenizer. Each residue is a token, `<mask>` hides
  one, and `<cls>` and `<eos>` mark the ends.
- **`ReadFasta` / `ParseFasta` / `WriteFasta`**: FASTA files, with multi-line sequences,
  comments and Windows line endings.

| Checkpoint | Layers | Width | Parameters | Size |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 6 | 320 | 8M | 31 MB |
| `esm2_t12_35M_UR50D` | 12 | 480 | 35M | 140 MB |
| `esm2_t30_150M_UR50D` | 30 | 640 | 150M | 600 MB |
| `esm2_t33_650M_UR50D` | 33 | 1280 | 650M | 2.6 GB |

## Predicting a hidden residue

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_sequences.hpp"

using namespace pulsatrix;

CPUBackend backend;
auto model = LoadEncoderLM("esm2_t33_650M_UR50D", &backend);
TextTokenizer tok = LoadEsmTokenizer("esm2_t33_650M_UR50D/vocab.txt");

// Ubiquitin with residue 11 (a lysine) hidden.
Encoding e = tok.encode("MQIFVKTLTG<mask>TITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG");
std::vector<float> ids(e.ids.begin(), e.ids.end());
Tensor logits = model->forward(Tensor(Shape({1, (int64_t)ids.size()}), &backend, ids));
// logits is (1, L, 33). Position 0 is <cls>, so residue 11 is position 11.
```

The 650M model puts lysine first, which is correct; the 8M model guesses glutamate. On a CPU,
the 650M model takes about 6 seconds to load and run.

Download a checkpoint with any Hugging Face client, or fetch three files directly:

```bash
for f in config.json model.safetensors vocab.txt; do
  curl -fsSL https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/$f -o esm2_t6_8M_UR50D/$f
done
```

## Reading the model's representations

After `forward()`:

- **`last_hidden_state()`** is `(N, L, width)`: the per-residue representations usually called
  "ESM embeddings", taken after the final LayerNorm. Average over the residues, without `<cls>`
  and `<eos>`, for a per-protein vector.
- **`hidden_states()`** holds the embeddings, then every layer's output. Probes and sparse
  autoencoders read these. Early and middle layers often work better than the last.
- **`layer(i).mha().last_attention_weights()`** is `(N, heads, L, L)`, each layer's attention.
  Some heads track which residues touch in the structure (contacts, roadmap PLM-4).

**Batches of different lengths.** Pad each sequence with `<pad>` (id 1) and pass a mask:

```cpp
model->set_padding_mask(keep);  // (N, L): 1 for a real token, 0 for padding
Tensor logits = model->forward(ids);
```

Padding is kept out of attention and out of token dropout's count, as in Hugging Face's
`attention_mask`. `clear_padding_mask()` removes it.

## Details that matter for correctness

- **Token dropout.** ESM-2 was trained with masked residues' embeddings zeroed. To match, every
  embedding is scaled by `(1 - 0.12) / (1 - masked fraction)`, even when nothing is masked. Its
  outputs depend on this.
- **Stored RoPE frequencies.** The checkpoints store their rotary frequencies rounded to fp16
  (0.316162 rather than 0.316228). The model was trained with those values, and transformers
  uses them, so pulsatrix reads them from the checkpoint. Recomputing them exactly shifts every
  attention map by about 2e-4.
- **Checkpoint names.** Older checkpoints name LayerNorm parameters `weight`/`bias`;
  transformers 5 saves `gamma`/`beta`. Both load.
- **Tokenizer quirks**, kept for parity with `EsmTokenizer`:
  - whitespace is dropped;
  - lowercase and unknown letters aren't residues;
  - a run of them becomes a single `<unk>` (`mkta` is one token).

  Uppercase your sequences first.

## Training and explaining

`EncoderLM` is an ordinary `Module`, so the rest of the library applies:

- `backward()` gives gradients for every parameter, and `set_requires_grad` freezes parts for
  fine-tuning.
- `TokenCrossEntropyLoss` with ignore-index -100 is the masked-LM loss.
- `propagate_relevance()` explains a logit with LRP and returns relevance per input token.
  `propagate_relevance_by_layer()` stops at every layer, which is what attribution graphs use.
- The masking step for masked-LM training, fine-tuning heads, and checked residue-level
  explanations are roadmap items PLM-7 and PLM-6.

## Checking against Hugging Face

`tools/golden/make_esm_golden.py` records transformers' outputs for a model:
- logits;
- every layer's hidden state;
- every attention map;
- a padded batch;
- the tokenizer's ids on edge cases.

`tests/encoder_lm_test.cpp` compares pulsatrix with them. CI runs it on a tiny random ESM-2
(`tests/fixtures/hf_tiny/esm`). Setting `PULSATRIX_GOLDEN_DIR` to a directory of downloaded
models checks those too. Worst differences on the real checkpoints (transformers 5.18, float32):

| Model | Hidden states (relative) | Attention (absolute) | Logits (relative) |
|---|---|---|---|
| ESM-2 8M | 2.8e-6 | 4.4e-6 | 2.7e-6 |
| ESM-2 650M | 3.7e-6 | 1.3e-5 | 3.6e-6 |

That is float32 precision: transformers' own float32 run differs from float64 by about 3e-6 in
attention. The tokenizer matches `EsmTokenizer` on every case recorded.
