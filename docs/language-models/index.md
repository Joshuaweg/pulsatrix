# Language Models

Pulsatrix can load a pretrained language model from the Hugging Face Hub and work with it in
plain C++: tokenize text, generate continuations, and explain any prediction token by token or
word by word, with no Python at runtime. Every step is checked against the reference
implementations: `transformers` for the logits, `tokenizers` for the token ids, and LXT for the
explanations.

This page walks through the whole path. The examples run as written on SmolLM2-135M.

## What you can do

- **Load a model** from a downloaded Hub directory: `LoadCausalLM(dir, &backend)`.
- **Tokenize** with the model's own `tokenizer.json`: `LoadTokenizerJson(path)`. You get ids,
  token strings, and byte offsets back into your text.
- **Generate**, greedily or by sampling (temperature, top-k, top-p, min-p, repetition penalty),
  with a KV cache: `Generate(model->next_token_logits(max_length), ids, config)`.
- **Explain a prediction** with AttnLRP (attention-aware Layer-wise Relevance Propagation). You
  get one relevance score per input token, which can be merged into words and drawn as a figure.
  `pulsatrix_explain_text` does all of this in one command.
- **Look inside the model**: read or replace its residual stream and MLPs with hooks, find
  features, diff a model against its fine-tune, steer it, and trace attribution graphs of its
  predictions (see [Look inside the model](#4-look-inside-the-model)).
- **Train or fine-tune**: `CausalLM` is an ordinary `Module`, so `backward()`, optimizers and
  checkpoints work on it.
- **Score answers and train on them**: read prompt/completion pairs from JSONL, batch them with
  the prompt masked out of the loss, and score log P(completion | prompt) or the probability of
  " Yes" against " No" (see [Score and fine-tune on answers](#5-score-and-fine-tune-on-answers)).
- **Generate with a diffusion language model** (MDLM), which fills a masked sequence over a
  number of steps, and keep the whole trajectory for explaining it (see
  [Diffusion language models](#diffusion-language-models)).
- **Check your setup** against Hugging Face with three command-line tools (see
  [Checking against Hugging Face](#checking-against-hugging-face)).

## Supported models

Decoder models in the Llama family load: Llama, SmolLM2, Qwen2/2.5, Qwen3 and Gemma 3. That
covers:

- grouped-query attention, QKV biases, QK-Norm, and tied or untied output layers
- Llama 3's RoPE scaling
- Gemma 3's sliding-window layers, with their own RoPE base, its GeGLU MLP, its `(1 + w)` and
  sandwich norms, and its scaled embeddings

These five have been checked end to end:

| Model | Logits vs `transformers` (CPU) | AttnLRP vs LXT | Tokenizer vs `tokenizers` |
|---|---|---|---|
| SmolLM2-135M | within 4.6e-4 | identical (largest difference 4e-6) | 10,000 of 10,000 lines identical |
| Qwen2.5-0.5B | within 1.1e-3 (6e-5 relative) | identical (8e-6) | 10,000 of 10,000 |
| Qwen3-0.6B | within 1.6e-4 | identical (1.2e-5) | 10,000 of 10,000 |
| Llama-3.2-1B | within 1.5e-4 | identical (2.3e-5) | 10,000 of 10,000 |
| Gemma 3 270M | within 1.6e-4 | identical (5.5e-5) | 10,000 of 10,000 |

"Identical" means a correlation of 1.000000 with LXT, and per-token relevance that matches to the
precision shown. The tokenizer check runs over a multilingual corpus: the Universal Declaration
of Human Rights in about 530 languages, plus code, numbers, unusual whitespace and emoji. The
tokenizers of gpt-oss, TinyLlama and Mistral 7B also match exactly, although those models aren't
checked yet. Greedy generation is token-for-token identical to `transformers`' `generate()`
on SmolLM2-135M and Gemma 3 270M.

**Not supported yet.** Loading a model with features pulsatrix can't run fails with a message
that names them:
- Gemma 1 and 2, and logit softcapping
- mixture-of-experts models
- RoPE scaling other than Llama 3's and linear (for example YaRN)

Nothing is silently approximated.

## 1. Get a model

Download a model's configuration, weights and tokenizer from the Hub with the `hf` command
(install it with `pip install huggingface_hub`):

```sh
hf download HuggingFaceTB/SmolLM2-135M --include "*.json" --include "*.safetensors" \
    --local-dir models/SmolLM2-135M
```

The directory then holds `config.json`, `model.safetensors` (or several shards with an index)
and `tokenizer.json`, which is all pulsatrix needs. The weights are usually stored as bf16 and
are widened to fp32 exactly on load.

## 2. Tokenize and generate

```cpp
#include <cstdio>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/generation.hpp"
#include "pulsatrix/tokenizer_json.hpp"

using namespace pulsatrix;

int main() {
    CPUBackend backend;
    TextTokenizer tokenizer = LoadTokenizerJson("models/SmolLM2-135M/tokenizer.json");
    std::unique_ptr<CausalLM> model = LoadCausalLM("models/SmolLM2-135M", &backend);

    Encoding prompt = tokenizer.encode("The three primary colors are");
    GenerationConfig config;
    config.max_new_tokens = 12;
    config.eos_token_ids = model->config().eos_token_ids;
    GenerationResult out = Generate(model->next_token_logits(256), prompt.ids, config);

    std::printf("%s\n", tokenizer.decode(out.tokens, /*skip_special_tokens=*/true).c_str());
}
```

```text
The three primary colors are red, yellow, and blue.

The color wheel
```

- `next_token_logits(256)` returns a generator with a KV cache for up to 256 positions, so each
  new token costs one position rather than the whole sequence.
- `GenerationResult` also has the new tokens alone (`new_tokens`) and each one's
  log-probability under the model (`log_probs`).
- To sample, set `config.do_sample = true` and any of `temperature`, `top_k`, `top_p`, `min_p` and
  `repetition_penalty`. The names and the order they're applied in follow Hugging Face's
  `GenerationConfig`. A `seed` makes sampling reproducible.

On a 16-core desktop CPU (AMD Ryzen AI Max+ 395) this program runs in about 1 second for
SmolLM2-135M and 3.5 seconds for Qwen2.5-0.5B, model loading included.

## 3. Explain a prediction

The quickest way is the `pulsatrix_explain_text` tool. It tokenizes your text, predicts the
next token, explains that prediction with AttnLRP, and writes a figure:

```sh
pulsatrix_explain_text models/SmolLM2-135M "The Eiffel Tower is located in the city of" --words -o paris.svg
```

SmolLM2 predicts " Paris", and the figure colors each word by how much it supported that
prediction:

| The | Eiffel | Tower | is | located | in | the | city | of |
|---|---|---|---|---|---|---|---|---|
| −1.598 | **+4.092** | −0.092 | +0.057 | +0.033 | +0.304 | −0.089 | −0.029 | +0.298 |

Red means a word pushed the model toward " Paris" and blue means it pushed away. Leave out
`--words` to see individual tokens instead: "Eiffel" is two tokens, "E" and "iffel", and nearly
all of its relevance sits on "iffel". Other options:

- `-o out.json` writes the data instead of a figure, for `pulsatrix_svg` or your own code.
- `--target " London"` explains a token of your choice instead of the most likely one.
- `--split whitespace` makes words the runs between spaces.
- `--device hip` runs on an AMD GPU.

The same steps in code:

```cpp
#include "pulsatrix/attnlrp_parity.hpp"   // LxtAttnLrpConfig
#include "pulsatrix/word_scores.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "pulsatrix/viz/text_relevance.hpp"

const std::string text = "The Eiffel Tower is located in the city of";
Encoding e = tokenizer.encode(text);
const int64_t L = static_cast<int64_t>(e.size()), V = model->config().vocab_size;
Tensor logits = model->forward(Tensor(Shape({1, L}), &backend, std::vector<float>(e.ids.begin(), e.ids.end())));

// Seed relevance at the predicted token's logit, at the last position.
std::vector<float> all = logits.to_host_vector();
const float* last = all.data() + (L - 1) * V;
const int64_t next = std::max_element(last, last + V) - last;
std::vector<float> seed(all.size(), 0.0f);
seed[(L - 1) * V + next] = last[next];

std::vector<float> relevance =   // one score per token
    model->propagate_relevance(Tensor(logits.shape(), &backend, seed), LxtAttnLrpConfig()).to_host_vector();

WordScores words = AggregateToWords(text, e, relevance);          // per word (sum)
TokenRelevanceDocument doc = MakeWordRelevanceDocument(text, words, "attn_lrp", tokenizer.decode({next}));
std::string svg = RenderTokenStripSvg(doc);                      // the figure
```

- **Relevance** follows LXT's AttnLRP rules, and pulsatrix's numbers match LXT's own
  implementation. The total isn't the logit: AttnLRP's attention rule doesn't conserve
  relevance. Read the scores relative to each other.
- **Words** come from the text, not the tokenizer, so every model gives the same words. The
  [data pipeline guide](../data-pipeline/index.md#per-word-scores) explains how a token's score
  is shared between words and the four ways to combine scores (`Sum`, `Mean`, `Max`, `MaxAbs`).
- **Figures and documents**: the [visualization guide](../visualization/documents.md#token-relevance-for-text)
  covers the token and word documents, the SVG strip and the `TokenRelevanceView` window widget.

## 4. Look inside the model

Two hooks give access to the model's internals during `forward()`:
- **`set_hidden_state_hook`** sees the residual stream at every position (0 the embeddings, i the
  output of block i).
- **`layer(i).set_mlp_hook`** sees each MLP's input and output.

Return the tensor unchanged to read it, or return a replacement to change what the model
computes. The [Mechanistic Interpretability](../mechanistic-interpretability/index.md) section
builds on them:

| To | Use | Guide |
|---|---|---|
| Compare a layer across models or checkpoints, or plot it in 2D | `LinearCka`, `MutualKnnAlignment`, `FitPca` | [Analyzing representations](../mechanistic-interpretability/representations.md) |
| Test whether a layer holds a concept you have labels for | `TrainLinearProbe`, `MeasureSelectivity` | [Probes and concepts](../mechanistic-interpretability/probes-and-concepts.md) |
| Collect activations, then find the features a layer uses | Hooks, then a TopK sparse autoencoder | [Finding features](../mechanistic-interpretability/featurizers.md) |
| See what chat tuning changed | `pulsatrix_diff_lm` (a crosscoder between the two models) | [Model diffing](../mechanistic-interpretability/model-diffing.md) |
| Push the model's behavior along a direction | `SteeringHook`, `MeasureSteering` | [Steering](../mechanistic-interpretability/steering.md) |
| Trace which features carry a prediction | `pulsatrix_explain_text --graph --transcoders` | [Attribution graphs](../mechanistic-interpretability/attribution-graphs.md) |

## 5. Score and fine-tune on answers

`pulsatrix/lm_scoring.hpp` handles data in the shape of a question and its answer:

```cpp
#include "pulsatrix/lm_scoring.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

// Each line: {"prompt": "... Is a launch in flight? A:", "completion": " No"}
const auto rows = ReadJsonl("train.jsonl");
const auto examples = TokenizePromptCompletions(rows, tokenizer);

// Training: only the completion's tokens count in the loss.
const CausalLmBatch batch = CollatePromptCompletions(examples, &backend);
TokenCrossEntropyLoss loss(&backend);
const float value = loss.forward(model->forward(batch.ids), batch.targets);
(void)model->backward(loss.backward());

// Scoring: log P(completion | prompt), or the choices' probabilities after each prompt.
const std::vector<double> log_p = CompletionLogProbs(*model, examples, &backend);
const int64_t yes = SingleTokenId(tokenizer, " Yes"), no = SingleTokenId(tokenizer, " No");
const ChoiceScores s = ScoreChoices(*model, prompts, {yes, no}, &backend);
const double log_odds = s.choice_log_probs[0][0] - s.choice_log_probs[0][1];
```

- **Masking:** position i of a batch predicts token i + 1. Every target outside the completion is
  `kIgnoreIndex`, so the loss is the completion's mean log loss per token.
- **Padding:** batches are right-padded. Causal attention keeps the padding from reaching any real
  position, so a padded batch scores exactly as each example does alone.
- **Choices:** `ScoreChoices` gives each choice's log probability over the whole vocabulary,
  renormalized over the choices, and the probability the choices take together. A model trained
  to answer " Yes" or " No" puts nearly all its mass on the two.
- **Memory:** the scorers gather only the logit rows they need on the device before copying them
  back. A batch's full logits are (N, L, vocab), about 400 MB for SmolLM2 at N = 8 and L = 256.
- **Speed:** a full fine-tuning step of SmolLM2-135M on a Radeon 8060S takes about 1.5 s for 8
  sequences of 256 tokens and 7.8 s for 8 of 512. Attention's cost grows with the square of the
  length, so keep prompts short.

## Diffusion language models

A masked diffusion language model doesn't write left to right. It starts from a sequence of mask
tokens, with a prompt if you have one, and fills it in over a number of steps, any position in
any order. At each step a bidirectional encoder predicts every masked position, and the sampler
commits some of those predictions. Pulsatrix loads MDLM checkpoints, such as
`kuleshov-group/mdlm-owt` (170M parameters, GPT-2's tokenizer), with the same `LoadEncoderLM`
that loads ESM-2. `SampleMaskedDiffusion` generates from them and records every step, so a
generation can be explained afterwards ([LRP for diffusion models](../interpretability/diffusion-lrp.md)).

```cpp
#include <algorithm>
#include <cstdio>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/masked_diffusion.hpp"
#include "pulsatrix/tokenizer_json.hpp"

using namespace pulsatrix;

int main() {
    CPUBackend cpu;
    auto model = LoadEncoderLM("mdlm-owt", &cpu);                   // config.json, model.safetensors
    TextTokenizer tok = LoadTokenizerJson("gpt2/tokenizer.json");  // MDLM uses GPT-2's tokenizer
    const int64_t mask = model->config().mask_token_id;

    std::vector<int64_t> seq(1024, mask);  // MDLM-owt was trained on 1024-token sequences
    const std::vector<int64_t> prompt = tok.encode("The city council voted", false).ids;
    std::copy(prompt.begin(), prompt.end(), seq.begin());  // a fixed prefix to continue

    MaskedDiffusionConfig config;
    config.steps = 128;
    config.mask_token_id = mask;
    config.schedule = UnmaskingSchedule::Ancestral;  // MDLM's own sampler
    config.do_sample = true;
    config.seed = 1;
    model->set_keep_activations(false);  // sampling needs only the outputs
    MaskedDiffusionResult out = SampleMaskedDiffusion(MakeDenoiser(*model, &cpu), seq, config);
    std::printf("%s\n", tok.decode(out.tokens).c_str());
    // out.steps[i].input is what the model saw at step i, and out.steps[i].writes what it filled.
}
```

There are two schedules for choosing which positions to fill:

- **`Confidence`** (LLaDA's) fills a fixed share of the masked positions at every step, taking
  the ones whose predicted token the model is surest of. Greedy, it is fully deterministic, which
  makes it the one to explain.
- **`Ancestral`** (MDLM's) fills each masked position independently at random, with the
  probability the noise schedule gives. Steps that fill nothing don't run the model.

Neither re-masks a filled position. Each step's record holds the exact input the model saw, so
rerunning the model on it reproduces that step.
`ExplainCommitments` (`masked_diffusion_lrp.hpp`) uses that to explain generated tokens back to
the prompt through every step that led to them; see
[D2 in pulsatrix](../interpretability/diffusion-lrp.md#d2-in-pulsatrix).

**How MDLM loads.** MDLM's blocks are DiT blocks with adaLN-Zero conditioning on the noise level,
but MDLM-owt was trained with that conditioning switched off, so every modulation vector is a
constant. `LoadEncoderLM` computes them once and folds them into the LayerNorms and output
projections, which is exact. The loaded model is an ordinary `EncoderLM`, so backward passes,
LRP and hooks all work on it. Against the authors' own modeling code, run on the CPU in float32,
logits match within 3.8e-6 and hidden states within 7.1e-6 (relative). A tiny model with random
modulation weights, which tests the folding itself, matches within 1e-5
(`tools/golden/make_mdlm_golden.py`, then `tests/mdlm_test.cpp`).

**Length and speed.** MDLM-owt was only ever trained on full 1024-token sequences. Sampled at
that length (128 ancestral steps), it writes fluent web text. Sampled at 64 tokens, it writes
nonsense, so keep the sequence at its trained length and put your prompt inside it. Every step is
a full forward pass over the whole sequence: about 8.6 s per step at 1024 tokens on the 16-core
test machine's CPU, so 128 steps take about 18 minutes. Use a GPU backend for real sampling.

**Not supported yet.**
- Models whose conditioning varies (`time_conditioning`, such as SEDD) fail to load with a message
  that says so. They need an adaLN layer, which is DIFF-6 on the roadmap.
- There's no remasking sampler (DPLM's) and no block-by-block, semi-autoregressive decoding.

## Checking against Hugging Face

Three tools compare pulsatrix with the reference libraries on your own machine. They need
reference files that small Python scripts write once, with `torch`, `transformers`, `tokenizers`
or `lxt` installed. The pinned environment is `tools/lrp_reference.Dockerfile`.

| Check | Write the reference (Python) | Compare (pulsatrix) | Passes when |
|---|---|---|---|
| Logits | `tools/golden/make_golden.py --model HuggingFaceTB/SmolLM2-135M --out golden/SmolLM2-135M` | `pulsatrix_golden golden/SmolLM2-135M` | every logit is within 1e-3, relative to the largest logit at its position |
| Explanations | `tools/golden/make_attnlrp_reference.py golden/SmolLM2-135M` | `pulsatrix_attnlrp golden/SmolLM2-135M` | per-token and per-KV-head relevance correlate above 0.99 with LXT |
| Tokenizer | `tools/tokenizers/make_tokenizer_reference.py reference tokenizer.json corpus.jsonl ref.jsonl` | `pulsatrix_tokenizer_parity tokenizer.json ref.jsonl` | ids and decoded text match on every line |

`make_golden.py` downloads the model at a pinned revision, so its output directory is a complete
model directory too. Setting `PULSATRIX_GOLDEN_DIR` (or `PULSATRIX_TOKENIZER_DIR`) to a directory of
such models makes the test suite check all of them.

## Performance and memory

- **CPU**: matrix multiplies use every core. On the 16-core test machine:

  | | Load | Forward pass (11 tokens) | Explanation |
  |---|---|---|---|
  | SmolLM2-135M | 0.7 s | 0.08 s | 0.3 s |
  | Llama-3.2-1B | 33 s | 0.6 s | 0.5 s |

  Loading large models is slow today. Speeding it up is a known follow-up, and once a model is
  loaded, generation and explanation are fast.
- **GPU**: on AMD GPUs (ROCm), pass a `HIPBackend` instead of the `CPUBackend`, or
  `--device hip` to the tools. The four models above were checked on a Radeon 8060S. The CUDA
  backend implements the same operations and builds in CI, but hasn't been run on these models.
- **Memory**: weights are fp32, and every parameter also has a gradient buffer, so plan on about
  14 bytes per parameter today. In practice that's 1.9 GB for SmolLM2-135M, 6.8 GB for
  Qwen2.5-0.5B and about 10 GB for Llama-3.2-1B, which peaks at 17 GB while explaining.
  Allocating gradients only when training is a planned improvement.

## Limits to know about

- **Batch size one:** generation and explanation work on a single sequence. Beam search, stop
  strings and batched generation aren't included yet.
- **Offsets:** tokenizer offsets are byte offsets into your text, where Hugging Face uses
  characters. Where NFC normalization merged characters, pulsatrix's offsets cover the merged
  characters fully, and Hugging Face's cover only the first.
- **Python:** these features are C++ only for now. The Python bindings cover tensors, layers and
  the explainers, but not tokenizers or `CausalLM`.

## Recipes and references

- [LRP guide: LXT on whole language models](../interpretability/lrp.md#validation-against-zennit-and-lxt)
- [Deep learning: Hugging Face checkpoints](../deep-learning/index.md#hugging-face-checkpoints)
  (config parsing, sharded weights, `LoadWeights` for other layouts)
- [Data pipeline: tokenizers](../data-pipeline/index.md#tokenizing-for-a-hugging-face-model)
- API reference: [`CausalLM`](../api/classpulsatrix_1_1CausalLM.html),
  [`TextTokenizer`](../api/classpulsatrix_1_1TextTokenizer.html)
