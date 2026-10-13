# Performance

How well does the model do, and where does it fail? This is the Performance question of the XAI
question bank, and an explanation can't answer it: an attribution says why the model made a
prediction, not whether the prediction is right. Pulsatrix answers it with the metrics in
`pulsatrix/performance.hpp` (KS-3):

- **`EvaluateClassifier`:** accuracy, log loss, the Brier score, calibration with a reliability
  diagram, a confusion matrix, and each class's precision, recall and F1;
- **`EvaluateRegressor`:** mean squared and absolute error, and R²;
- **`PerSliceError`:** the error on each slice of the data, each tested against the rest.

Every number matches scikit-learn's on the same data, and the slice test matches SciPy's
`fisher_exact`. `tests/performance_test.cpp` checks them against values from
`tools/generate_performance_reference_values.py`.

## Scoring a classifier

`EvaluateClassifier` takes the model's probabilities, `(N, K)`, and the true classes:

```cpp
#include "pulsatrix/performance.hpp"

Tensor probabilities = softmax.forward(model.forward(inputs));   // (N, K)
ClassificationReport r = EvaluateClassifier(probabilities, labels);
std::printf("accuracy %.3f, log loss %.3f, Brier %.3f, ECE %.3f\n", r.accuracy, r.log_loss, r.brier, r.ece);
```

The predicted class is the most probable one, the lower index on a tie. Rows aren't renormalized.
The definitions are scikit-learn's:

| Field | Definition | scikit-learn |
|---|---|---|
| `accuracy` | the fraction of inputs whose predicted class is the label | `accuracy_score` |
| `log_loss` | the mean of −log p(label); a probability of 0 counts as 10⁻¹⁵ | `log_loss` |
| `brier` | the mean over inputs of Σₖ (pₖ − yₖ)², from 0 to 2 (Brier, 1950) | `brier_score_loss(scale_by_half=False)` |
| `precision`, `recall`, `f1` | per class; 0 where there is nothing to divide by | `precision_recall_fscore_support(zero_division=0)` |
| `confusion[true][predicted]` | counts | `confusion_matrix` |

## Calibration

A model is calibrated when its confidence means what it says: of the predictions it makes with
probability 0.8, 80% are right. Modern networks are often overconfident (Guo, Pleiss, Sun and
Weinberger, ICML 2017).

`EvaluateClassifier` groups the predictions into equal-width bins of confidence, the largest
probability: 15 by default, as Guo et al. used (`ClassificationOptions::bins`). Each bin is
(lower, upper]. The report gives:

- **`bins`:** each bin's count, mean confidence and accuracy, the reliability diagram;
- **`ece`**, the expected calibration error: the gap between accuracy and confidence, averaged over
  the bins and weighted by their size;
- **`mce`**, the maximum calibration error: the largest gap in a non-empty bin.

On the test data, 200 predictions from 4 classes:

```
accuracy 0.575  log loss 1.411  Brier 0.662  ECE 0.152  MCE 0.455
(0.533, 0.600]  n=28  confidence 0.567  accuracy 0.393
(0.733, 0.800]  n=17  confidence 0.769  accuracy 0.471
(0.800, 0.867]  n=28  confidence 0.834  accuracy 0.536
(0.867, 0.933]  n=23  confidence 0.904  accuracy 0.696
```

Its most confident predictions are right far less often than it claims.

!!! note
    The ECE depends on the number of bins, and with few predictions per bin it is biased upward.
    Compare models with the same bins, and read the reliability diagram, not just the number.

## Error by slice

A model can do well on average and fail one group. `PerSliceError` reports the error on each
slice, such as a demographic group, a data source or a class of input. It tests each slice against
the rest of the data, so a failure confined to one group isn't averaged away:

```cpp
SliceErrorReport s = PerSliceError(probabilities, labels, slices);   // slices: one name per input
std::cout << DescribeSliceError(s);
```

```
error 0.425 over 200 inputs; each slice against the rest (Fisher's exact test, alpha 0.05 over 3 slices)
  slice  n       error    95% interval      rest     p
  c      66      0.591    0.470 to 0.701    0.343    0.000751  WORSE
  a      67      0.358    0.254 to 0.478    0.459    0.935
  b      67      0.328    0.228 to 0.447    0.474    0.983
at least one slice is worse than the rest
```

How each slice is tested:

- **Errors of 0 or 1** (right or wrong) use Fisher's exact test, one-sided. Is the slice wrong
  more often than the rest? The interval is Wilson's score interval.
- **Losses** use a permutation test of the slice's mean against the rest's: 999 relabelings that
  keep the slice sizes. The interval is the normal approximation. Pass the losses as a vector:
  `PerSliceError(errors, slices)`.
- **A slice is worse** when its p-value is at most α divided by the number of slices: Bonferroni's
  correction, so testing many slices doesn't find one by chance. α is 0.05 by default
  (`SliceErrorOptions::alpha`); the intervals are at level 1 − α.

A slice that isn't flagged hasn't been shown to be fine. A small slice can have a high error and
still not be significantly worse; its interval shows how little is known.

## In a report

The Control reason requires per-slice error. The planner marks it available when the caller has
the dataset's labels and slices (`ModelTraits::has_labels`, or `pulsatrix_plan --labels`).
`BuildExplanation` then runs it on `request.dataset`, `request.labels` and `request.slices`. The
check fails, and blocks the report, when a slice is worse than the rest:

```cpp
request.reason = XaiReason::Control;
request.traits.has_dataset = true;
request.traits.has_labels = true;
request.dataset = inputs;      // each (1, ...)
request.labels = labels;
request.slices = slices;
Explanation e = BuildExplanation(request, x, explain, predict, &model);
```

For a regressor, or any model whose error isn't the wrong class, run `PerSliceError` on its losses
and pass `SliceErrorCheck(report)` to `ReportExplanation`.

The Performance question itself plans `ClassifierMetrics`, `RegressorMetrics` and
`PerSliceError`. They describe how the model behaves, not what it learned, so the planner doesn't
ask for a null-model baseline for them.

Each report has a JSON document: `pulsatrix.classification_report.v1`,
`pulsatrix.regression_report.v1` and `pulsatrix.slice_error.v1`.
