# Recipe: Is the Student Born Better?

**What you'll build:** born-again networks (Furlanello et al., ICML 2018) on MNIST, with the
controls their claim needs. A network is trained on the labels (generation 0). A second network
of the same size is trained on the first one's outputs (generation 1), a third on the second's,
and a fourth on the third's. Furlanello et al. found students that beat their teachers, and an
ensemble of the generations that did better still. The recipe repeats this over five seeds, against
three controls:

- **another network trained on the labels** from another start, so a gain can be set against the
  spread between two runs that differ only in their seed;
- **the permuted teacher** (Furlanello et al.'s DKPP): generation 1 again, from the same start and
  with the same batches, but taught by generation 0's outputs with every class except the top one
  shuffled. It keeps the teacher's answer and confidence and scrambles what the teacher says about
  the other classes;
- **an ensemble of three networks trained on the labels,** the fair comparison for the ensemble of
  the three born-again generations.

Accuracy can't say whether a student uses its teacher's evidence. So the recipe also compares
explanations, with the test from KD-2: saliency maps of the true class on 200 test digits, from
generation 1 and generation 0, and from the other label network and generation 0.

The predictions were written down before the first run:

1. Generation 1 is more accurate than generation 0 on average, by less than one percentage point,
   and not on every seed.
2. The permuted teacher's student is less accurate than generation 1: the gain comes from what
   the teacher says about the wrong classes.
3. The ensemble of three label networks is at least as accurate as the ensemble of three
   born-again generations, which learned from one another and so make more of the same errors.
4. Generation 1's explanations agree with generation 0's more than the other label network's do,
   on every seed.

Only prediction 3 holds. Generation 1 is no better than its teacher. The permuted teacher's
student is better than generation 1 on every seed. And generation 1's explanations are no closer
to its teacher's than an independent network's. The teacher is 99% sure on the images it trained
on, so a student trained to match it at T = 1 is close to being trained on the labels again. A
follow-up with a softer teacher (T = 4) separates the two questions: the student copies more of its
teacher's evidence, and is less accurate for it.

CMake target: `born_again_networks_recipe` (`examples/recipes/born_again_networks.cpp`).

```bash
./build/born_again_networks_recipe MNIST_DIR out/ [--device cpu|hip] [--seeds 5] [--train 10000] \
    [--epochs 10] [--generations 3] [--temperature 1] [--smoke]
```

!!! note
    The recipe needs MNIST's four IDX files in `MNIST_DIR` (`tools/fetch_mnist.py` downloads
    them). It runs on the CPU in a few minutes. `--smoke` runs on made-up data in a few seconds to
    check the plumbing; its numbers mean nothing.

## Code

Each network is LeNet-300-100 (784 → 300 → 100 → 10, with ReLUs), trained with Adam for 10 epochs
on the first 10,000 training images. Generation 0 learns from the labels, and each later
generation from its teacher's logits, precomputed on every training image:

```cpp
DistillationOptions dopt;
dopt.temperature = o.temperature;           // 1: Furlanello et al.'s choice
DistillationLoss soft(backend, dopt);       // no labels: alpha doesn't apply
(void)soft.forward(net.seq.forward(x), teacher_logits_of_the_batch);
(void)net.seq.backward(soft.backward());
```

The permuted teacher is the same loop with the teacher's logits shuffled again each epoch:

```cpp
targets = PermuteDarkKnowledge(teacher_logits, seed * 1000 + epoch, &cpu);
```

The ensembles average the members' probabilities:

```cpp
const Tensor ensemble = EnsembleLogProbabilities({gen1_logits, gen2_logits, gen3_logits}, &cpu);
```

Each explanation is the gradient of the true class with respect to the pixels. Two networks'
maps of the same 200 digits are compared with `CompareExplanations` (KD-2), whose permutation test
asks whether the agreement is about these digits or just the general shape of a saliency map.

## Expected output

On the CPU:

```
Data: 10000 training and 10000 test images (MNIST)
Network: 784 -> 300 -> 100 -> 10, Adam lr 1e-3, 10 epochs, batches of 128; students at T = 1.0

seed 0: generations 95.62% 95.49% 95.47% 95.48%; other label network 95.75%, permuted teacher 95.70%; ensembles: born-again 95.65%, label 95.98%
seed 1: generations 95.71% 95.66% 95.46% 95.43%; other label network 95.56%, permuted teacher 95.73%; ensembles: born-again 95.79%, label 96.09%
seed 2: generations 95.65% 95.52% 95.57% 95.46%; other label network 95.49%, permuted teacher 95.69%; ensembles: born-again 95.68%, label 96.00%
seed 3: generations 95.45% 95.64% 95.50% 95.69%; other label network 95.56%, permuted teacher 95.89%; ensembles: born-again 95.83%, label 95.96%
seed 4: generations 95.72% 95.63% 95.61% 95.47%; other label network 95.74%, permuted teacher 95.63%; ensembles: born-again 95.72%, label 96.00%
(354 s)

Test accuracy, mean ± sd over 5 seeds:
  generation 0 (labels)             95.63% ± 0.11
  generation 1                      95.59% ± 0.08
  generation 2                      95.52% ± 0.07
  generation 3                      95.51% ± 0.10
  another network on the labels     95.62% ± 0.12
  student of the permuted teacher   95.73% ± 0.10
  ensemble of generations 1-3       95.73% ± 0.08
  ensemble of 3 label networks      96.01% ± 0.05

Generation 0's mean top probability on its training images, at T = 1.0: 99.07%

The predictions:
  1. generation 1 minus generation 0: -0.04 points, higher on 1 of 5 seeds (two label networks differ by -0.01 ± 0.14): not as predicted
  2. generation 1 minus the permuted teacher's student: -0.14 points, higher on 0 of 5 seeds: not as predicted
  3. label ensemble minus born-again ensemble: +0.27 points, higher on 5 of 5 seeds: as predicted
  4. explanations of 200 test digits (the gradient of the true class), Spearman with generation 0's:
       seed 0: generation 1 0.386 (p 0.001, CKA 0.929), the other label network 0.405 (p 0.001, CKA 0.935)
       seed 1: generation 1 0.379 (p 0.001, CKA 0.935), the other label network 0.391 (p 0.001, CKA 0.915)
       seed 2: generation 1 0.392 (p 0.001, CKA 0.934), the other label network 0.384 (p 0.001, CKA 0.922)
       seed 3: generation 1 0.389 (p 0.001, CKA 0.901), the other label network 0.383 (p 0.001, CKA 0.879)
       seed 4: generation 1 0.390 (p 0.001, CKA 0.924), the other label network 0.391 (p 0.001, CKA 0.922)
     generation 1 agrees more on 2 of 5 seeds: not as predicted
```

The run shared the machine with a test suite; alone, it takes about three minutes. Every seed's
numbers are the same from run to run.

## What's happening

| | Prediction | Result |
|---|---|---|
| 1 | Generation 1 beats generation 0, by under a point, not on every seed | **Fails:** −0.04 points, better on 1 of 5 seeds |
| 2 | The permuted teacher's student is worse than generation 1 | **Fails:** it is better, by 0.14 points, on 5 of 5 seeds |
| 3 | Three label networks ensemble at least as well as three generations | **Holds:** +0.27 points, on 5 of 5 seeds |
| 4 | Generation 1's explanations are closer to its teacher's | **Fails:** closer on 2 of 5 seeds, by at most 0.008 |

**No born-again gain.** Generation 1 is 0.04 points below generation 0, and two networks trained
on the labels from different starts differ by 0.14 points (standard deviation). Later generations
drift down a little, to 95.51%. The teacher is the likely reason: on the images it trained on, its
mean top probability is 99.07%. At T = 1 its outputs are nearly the labels, with about 1% of the
probability left to say anything about the other classes, so the student is trained on the labels
again, from a different start.

**The scrambled teacher did better than the real one,** on every seed (a sign test gives
p = 1/32 ≈ 0.03 for five of five). Shuffling keeps each row's values, so its targets are exactly
as confident as the real teacher's; only which wrong classes get the leftover probability changes.
One reading, not tested here: the real teacher's leftover probability repeats its own confusions,
which its students then copy, while shuffled probability is noise that regularizes. Either way,
this run's gain over the labels didn't come from the dark knowledge.

**The ensembles agree with prediction 3.** Three networks trained on the labels from different
starts ensemble to 96.01%, and three born-again generations to 95.73%. That fits generations that
copy one another and so make more of the same errors, though the recipe doesn't measure the
overlap directly. The born-again ensemble is no better than the single permuted-teacher student.

**The explanations agree with the accuracy.** Every pair of maps agrees far more than maps of
different digits do (p = 0.001, the smallest 999 permutations can give), and generation 1 agrees
with its teacher no more than an independent network does: Spearman 0.38 to 0.41 for both. A
student trained on near-labels didn't learn its teacher's evidence any more than a fresh network
learns it from the labels alone.

### A follow-up: a softer teacher

The first run points at the temperature: at T = 1 the teacher has almost nothing to say beyond the
labels. `--temperature 4` softens both distributions, so the teacher's mean top probability on its
training images drops from 99.07% to 71.72%. This run was designed after seeing the first, so it
follows up on it and isn't part of its test. Its predictions were written down before it ran:
(a) generation 1 is more accurate than generation 0 on average, and (b) generation 1's
explanations agree with its teacher's more than the other label network's do, on at least 4 of 5
seeds.

```
Test accuracy, mean ± sd over 5 seeds:
  generation 0 (labels)             95.63% ± 0.11
  generation 1                      95.30% ± 0.15
  generation 2                      95.05% ± 0.08
  generation 3                      94.85% ± 0.07
  another network on the labels     95.62% ± 0.12
  student of the permuted teacher   95.67% ± 0.10
  ensemble of generations 1-3       95.12% ± 0.14
  ensemble of 3 label networks      96.01% ± 0.05

Generation 0's mean top probability on its training images, at T = 4.0: 71.72%

The predictions:
  1. generation 1 minus generation 0: -0.33 points, higher on 0 of 5 seeds (two label networks differ by -0.01 ± 0.14): not as predicted
  2. generation 1 minus the permuted teacher's student: -0.37 points, higher on 0 of 5 seeds: not as predicted
  3. label ensemble minus born-again ensemble: +0.89 points, higher on 5 of 5 seeds: as predicted
  4. explanations of 200 test digits (the gradient of the true class), Spearman with generation 0's:
       seed 0: generation 1 0.461 (p 0.001, CKA 0.923), the other label network 0.405 (p 0.001, CKA 0.935)
       seed 1: generation 1 0.450 (p 0.001, CKA 0.927), the other label network 0.391 (p 0.001, CKA 0.915)
       seed 2: generation 1 0.464 (p 0.001, CKA 0.947), the other label network 0.384 (p 0.001, CKA 0.922)
       seed 3: generation 1 0.460 (p 0.001, CKA 0.955), the other label network 0.383 (p 0.001, CKA 0.879)
       seed 4: generation 1 0.459 (p 0.001, CKA 0.943), the other label network 0.391 (p 0.001, CKA 0.922)
     generation 1 agrees more on 5 of 5 seeds: as predicted
```

- **(a) fails, and more clearly than before.** Generation 1 is 0.33 points below its teacher, on
  every seed, and each generation loses more: 95.30%, 95.05%, 94.85%. The born-again ensemble falls
  to 95.12%, 0.89 points below the label networks' ensemble.
- **(b) holds.** Generation 1's saliency maps agree with its teacher's at Spearman 0.45 to 0.46, on
  every seed, against 0.38 to 0.41 for the independent network.

So the softer teacher passes on more of its evidence, and the student is less accurate for it.
Agreement between explanations measures how faithfully the student copies the teacher, not whether
the copy is any good. A student can agree more with a teacher's explanations and make more
mistakes. This also fits the T = 1 run: with nothing to pass on but the labels, the student matched
the teacher no better than a stranger did.

### A measure that couldn't fail

The first version of the recipe compared gradient × input maps, the usual choice. Every
comparison came out at Spearman 0.995: generation 1 against its teacher and the independent
network against it, on every seed. Gradient × input is zero wherever the pixel is zero, and most
of an MNIST digit is blank background, the same blank pixels in both maps. Two maps that share
most of their zeros have almost the same ranks whatever the networks do. Saliency, the gradient
alone, has no such shared zeros. Before trusting an agreement score, check that two maps you
expect to differ can come out different.

## See also

- [Knowledge distillation](../../deep-learning/index.md#knowledge-distillation): `DistillationLoss`,
  and the born-again pieces in `born_again.hpp`.
- [Half the layers](half_the_layers.md): distillation into a student half the depth of its
  teacher, on a language model.
