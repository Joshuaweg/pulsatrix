# Recipe: A Notebook Report from C++

**What you'll build:** a Jupyter notebook and an HTML page, written by a C++ program, that
document a small experiment. A small MLP learns which XOR quadrant a point of the unit square is
in, and the report holds:

- the text and the code;
- the training curve;
- a heatmap of what the model learned;
- an LRP explanation;
- the model's probabilities as a tensor.

Nothing needs Python, a Jupyter kernel or a download.

CMake target: `notebook_report_recipe` (`examples/recipes/notebook_report.cpp`).

```bash
./build/notebook_report_recipe out/      # writes out/xor_report.ipynb and out/xor_report.html
```

Open `xor_report.ipynb` in VS Code, JupyterLab or on GitHub, or `xor_report.html` in a browser.
JupyterLab asks for the notebook's C++ kernel; choose "No Kernel" and every output still shows.
See [Notebooks and Reports](../../visualization/notebooks.md) for what a report can hold.

## Code

```cpp
#include "pulsatrix/viz/report.hpp"

// ... train the model, keeping the loss and accuracy in a TrainingLogDocument ...
// ... compute the probability map (a HeatmapDocument) and the LRP relevance ...

Report report("XOR quadrants: training and explaining a small MLP");
report.markdown("A small MLP learns which XOR quadrant a point of the unit square is in ...");
report.code("for (int64_t step = 0; step <= 400; ++step) { ... }")  // shown, not run
      .show(log);                                                    // the training curve
report.text("final loss 0.0058, training accuracy 100.0%");
report.markdown("## What the model learned").show(landscape);       // a heatmap
report.code("Attribution r = LRP().explain(ctx, probe, /*target_index=*/1, &cpu);").show(why);
report.show(Tensor(Shape({4, 2}), &cpu, probabilities));            // a tensor summary
report.save_ipynb(out + "/xor_report.ipynb");
report.save_html(out + "/xor_report.html");
```

- `markdown()` and `code()` add cells. The code is shown as text and not run: the program
  computes everything itself.
- `show()` adds any value pulsatrix can display (a training log, a heatmap, an attribution, a
  tensor and more) under the cell before it. `text()` adds plain text.
- Every figure is computed from the trained model in the same program, so the report can't drift
  from the code that made it.

Full source:
[`examples/recipes/notebook_report.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/notebook_report.cpp).

## Expected output

```
trained: loss 0.0058, accuracy 100.0%
wrote out/xor_report.ipynb and out/xor_report.html (11 cells)
```

The program runs in well under a second. In the report:

- **The training curve** reaches 100% accuracy after about 120 of the 400 steps.
- **The probability map** shows the XOR pattern: class 1 in the top-left and bottom-right
  quadrants, with sharp boundaries at 0.5.
- **The LRP heatmap** explains class 1 at one point in each quadrant. At (0.2, 0.8), the high `y`
  is the evidence for class 1 (red). At (0.8, 0.2) it's the high `x`. At (0.2, 0.2) and
  (0.8, 0.8), the inputs count against it (blue).

The exact numbers can differ slightly between compilers.
