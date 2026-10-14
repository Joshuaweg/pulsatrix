# Recipe: Half the Layers

**What you'll build:** a student language model half the depth of its teacher, made the way
DistilBERT was (Sanh et al., 2019). SmolLM2-135M-Instruct has 30 blocks. The student keeps blocks
0, 2, 4, ..., 28, starts from their weights, and trains on chat text with DistilBERT's three
losses: the teacher's softened outputs, the next token, and a cosine loss pulling the student's
last hidden states toward the teacher's. Five students, the same 300 steps on the same batches:

- **untrained:** the 15 kept blocks as they are;
- **language modeling:** next-token cross-entropy alone;
- **distilled:** DistilBERT's 5 : 2 : 1 mix of distillation (T = 2), cross-entropy and cosine;
- **distilled without the cosine loss:** 5 : 2;
- **distilled from random blocks:** the 5 : 2 : 1 mix, with every block's weight matrices redrawn
  at random (each at its own scale). The embedding and the norms are still the teacher's.

Then a question the loss can't answer: does the distilled student predict the next token from the
same tokens as the teacher? AttnLRP token relevance for the teacher's top next token, compared
between teacher and student with KD-2's test.

The predictions were written down before the first run:

1. Half the layers, untrained, breaks the model: held-out perplexity more than 10 times the
   teacher's.
2. Distillation beats language modeling alone: lower held-out perplexity and more of the
   teacher's top-1 next tokens.
3. Starting from the teacher's blocks matters more than the loss: the student from random blocks
   ends with more than twice the language-modeling student's perplexity.
4. The cosine loss matters little here: dropping it moves held-out perplexity by less than 5%.
5. The distilled student's token relevance agrees with the teacher's more than the
   language-modeling student's does (mean Spearman).

Predictions 1, 3 and 4 hold; 2 and 5 fail. Distillation makes the student's *outputs* much more
like the teacher's (a third of the KL divergence) without making its *explanations* any more like
the teacher's. And the headline perplexity hides two different things, which the recipe separates.

CMake target: `half_the_layers_recipe` (`examples/recipes/half_the_layers.cpp`).

```bash
./build/half_the_layers_recipe MODEL_DIR out/ --text conversations.jsonl [--device hip] [--steps 300]
```

!!! note
    The recipe needs [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
    (270 MB): `model.safetensors`, `config.json` and `tokenizer.json` in one directory, and a JSONL
    file of chat conversations from `tools/explain/fetch_chat_sample.py`. On a GPU a run takes
    about 15 minutes. `--smoke` runs the tiny test model (`tests/fixtures/hf_tiny/llama`) on a few
    built-in sentences to check the plumbing; its numbers mean nothing.

## Code

The text is cut into windows of 128 tokens. Whole conversations go to training (the first 90%) or
to the held-out set, so none is in both. The student, and taps that give the cosine loss its
gradient:

```cpp
std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, EveryOtherLayer(30), backend);
student->set_intervention(14, std::make_unique<HiddenStateTap>(backend));
teacher->set_intervention(29, std::make_unique<HiddenStateTap>(backend));
```

One distillation step, with DistilBERT's weights: `DistillationLoss` with alpha = 5/7 mixes the
soft and hard terms 5 : 2, and the cosine loss enters at 1/7:

```cpp
DistillationOptions dopt;
dopt.temperature = 2.0f;
dopt.alpha = 5.0f / 7.0f;
DistillationLoss kd(backend, dopt);

const Tensor logits = student->forward(ids);
float loss = kd.forward(logits, teacher->forward(ids), targets);
loss += cos.forward(tap->hidden(), teacher_tap->hidden()) / 7.0f;
Tensor g = cos.backward();
backend->axpby(1.0f / 7.0f, g.data(), 0.0f, g.data(), g.data(), static_cast<size_t>(g.numel()));
tap->add_gradient(g);
student->backward(kd.backward());
ClipGradNorm(*student, 1.0f);
opt.step(*student);                                  // AdamW, lr 2e-4
```

The explanations are `DiffRelevance(teacher, student, ids, target, LxtAttnLrpConfig())` on 24
held-out prompts of 32 tokens, with the target fixed to the teacher's top next token, so both
models explain the same prediction.

## Expected output

On a Radeon 8060S (gfx1151):

```
Teacher: build-hip10-release/models/SmolLM2-135M-Instruct (30 blocks, hidden 576)
Students: blocks 0, 2, ..., 28 (15 blocks)
Text: 2393 training and 64 held-out windows of 128 tokens
Training: 300 steps of 8 windows, AdamW lr 2e-4, gradients clipped at 1

Teacher: held-out perplexity 6.08
Untrained student: held-out perplexity 1423820.74

Student: language modeling
    step 50: objective 2.833, held-out perplexity 12.41
    step 100: objective 1.933, held-out perplexity 7.50
    step 150: objective 1.882, held-out perplexity 5.97
    step 200: objective 1.618, held-out perplexity 5.20
    step 250: objective 1.561, held-out perplexity 4.86
    step 300: objective 1.530, held-out perplexity 4.55
    131 s

Student: distilled
    step 50: objective 4.251, held-out perplexity 14.97
    step 100: objective 2.431, held-out perplexity 7.83
    step 150: objective 2.125, held-out perplexity 6.11
    step 200: objective 1.876, held-out perplexity 5.41
    step 250: objective 1.698, held-out perplexity 4.92
    step 300: objective 1.805, held-out perplexity 4.60
    240 s

Student: distilled without the cosine loss
    step 50: objective 4.152, held-out perplexity 15.29
    step 100: objective 2.365, held-out perplexity 7.92
    step 150: objective 2.057, held-out perplexity 6.14
    step 200: objective 1.822, held-out perplexity 5.36
    step 250: objective 1.648, held-out perplexity 4.92
    step 300: objective 1.748, held-out perplexity 4.61
    240 s

Student: distilled from random blocks
    step 50: objective 6.490, held-out perplexity 48.92
    step 100: objective 4.358, held-out perplexity 22.71
    step 150: objective 4.090, held-out perplexity 19.24
    step 200: objective 3.919, held-out perplexity 16.90
    step 250: objective 3.437, held-out perplexity 15.35
    step 300: objective 3.902, held-out perplexity 15.05
    240 s

On 64 held-out windows (8128 predicted tokens; 59 of the windows start with the opening every conversation shares, 24 tokens):
  model                                perplexity   opening   rest    KL from teacher   teacher's top-1
  teacher                              6.08         229.87    2.94
  untrained                            1423820.74   1084399.79 1503845.96 13.031            0.2%
  language modeling                    4.55         1.00      6.15    2.814             61.1%
  distilled                            4.60         1.48      5.78    0.898             63.8%
  distilled without the cosine loss    4.61         1.41      5.85    0.910             65.1%
  distilled from random blocks         15.05        1.49      23.94   2.036             54.1%

Token relevance (AttnLRP) of the teacher's top next token, 24 prompts of 32 tokens after the shared opening, against the teacher's:
  untrained           Spearman 0.144 (p 0.877), top-25% overlap 0.391, CKA 0.045
  language modeling   Spearman 0.657 (p 0.001), top-25% overlap 0.677, CKA 0.453
  distilled           Spearman 0.624 (p 0.001), top-25% overlap 0.635, CKA 0.601

The predictions:
  1. untrained / teacher perplexity: 234066.1x: as predicted
  2. distilled vs language modeling: perplexity 4.60 vs 4.55, teacher's top-1 63.8% vs 61.1%: not as predicted
  3. random blocks / language modeling perplexity: 3.3x: as predicted
  4. dropping the cosine loss moves perplexity by 0.2%: as predicted
  5. relevance Spearman with the teacher, distilled vs language modeling: 0.624 vs 0.657: not as predicted
     prompt by prompt, the distilled student is the closer of the two on 7 of 24
```

The runs are deterministic: the same command gives the same numbers.

## What's happening

| | Prediction | Result |
|---|---|---|
| 1 | Untrained, perplexity over 10 times the teacher's | **Holds:** 234,066 times |
| 2 | Distilled beats language modeling on perplexity and the teacher's top-1 | **Fails:** perplexity 4.60 against 4.55; top-1 63.8% against 61.1% |
| 3 | Random blocks end over twice as perplexed as language modeling | **Holds:** 3.3 times |
| 4 | Dropping the cosine loss moves perplexity under 5% | **Holds:** 0.2% |
| 5 | Distilled relevance closer to the teacher's than language modeling's | **Fails:** Spearman 0.624 against 0.657; closer on 7 of 24 prompts |

**Taking out every other block breaks the model completely.** The untrained student's perplexity
is over a million, and it picks the teacher's top token 0.2% of the time. The blocks that are left
were never trained to follow one another. Yet 300 steps, about one pass over the training text,
turn every student that started from them back into a working model: perplexity 6.15 or less on
the text after the opening, against the teacher's 2.94.

**Students "beat" the teacher, and that isn't what it looks like.** Every trained student but one
has a lower held-out perplexity than the teacher: 4.55 against 6.08. All 2,379 conversations open
with the same 24 tokens, a system prompt naming the assistant, and 59 of the 64 held-out windows
start with them. The teacher predicts these tokens badly (perplexity 229.87), and the students
learn them within a few hundred steps (1.00 to 1.49). On the rest of each window, the teacher is
far better than any student: 2.94 against 5.78 at best. The split was added after the first run
showed the students ahead of their teacher. It explains a number; it isn't part of a test.

**So prediction 2 fails on the number it named, and the split shows why.** On the rest of the
text the distilled student is better than the language-modeling one (5.78 against 6.15), picks the
teacher's top token more often (63.8% against 61.1%), and is three times closer to the teacher's
whole distribution (KL 0.90 against 2.81). On the opening it is worse (1.48 against 1.00), because
it is partly trained toward a teacher that predicts the opening badly. The opening is a fifth of
the tokens in most windows, enough to tip the total.

**The teacher's weights matter more than any loss.** From random blocks, with the same losses and
steps, the student ends at 15.05, and at 23.94 on the rest of the text. **The cosine loss doesn't
matter here:** without it, perplexity moves 0.2%, and the top-1 agreement is even a little higher
(65.1%).

**The explanations don't follow the outputs.** Both trained students explain their predictions
much like the teacher: Spearman 0.62 and 0.66, where the untrained student is at 0.14 and no
better than its maps of other prompts (p = 0.877). But distillation doesn't bring the student's
relevance closer to the teacher's. The language-modeling student is closer on Spearman and on the
top 25% of tokens, and the distilled student is the closer one on only 7 of the 24 prompts. Only CKA, which compares the pattern across
prompts, favors the distilled student (0.601 against 0.453). Matching the teacher's output
distribution three times more closely didn't make the student use the teacher's evidence more. The
[born-again recipe](born_again_networks.md) found the reverse: a student that copied more of its
teacher's evidence and got less accurate. Output agreement and explanation agreement are separate
things, and each has to be measured.

### Two instruments that needed fixing

Both were found by reading the raw numbers of the first run, and both changes are in the code
above.

- **Perplexity over everything.** The first run printed only the overall perplexity, which put
  every trained student ahead of its teacher. Splitting off the shared opening showed why.
- **Explanation prompts that were all the same.** The first run explained the first 32 tokens of
  each window, so most prompts were the same 24-token system prompt plus 8 more tokens. The
  language-modeling student's 24 per-prompt agreements took only 7 distinct values. On those near-copies,
  prediction 5 held (Spearman 0.610 against 0.582). With each prompt starting after the shared
  opening, it fails (0.624 against 0.657). The fix went against the prediction. Check that a test's
  inputs really differ before reading its result.

## See also

- [Knowledge distillation](../../deep-learning/index.md#knowledge-distillation), and
  "A smaller language model, DistilBERT's way" in the same guide.
- [Is the student born better?](born_again_networks.md): distillation into a student the size of
  its teacher.
