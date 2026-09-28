# Recipe: Training dashboard

**What you'll build:** a live GUI window showing `XorNetwork` training in real time — a loss
curve that visibly animates as `ImPlotMetricsSink` logs each step and `TrainingDashboard`
draws the result.

This one is a GUI demo rather than a stdout-printing recipe target (drawing calls into
ImGui/ImPlot aren't meaningfully testable/scriptable headlessly — see the
[Visualization docs](../../visualization/index.md) for why), so it's cross-linked from the
existing demo instead of a separate recipe target, the same allowance
[Recipes](../index.md) makes for demos elsewhere.

CMake target: `training_dashboard_demo`
(`examples/viz/training_dashboard_demo.cpp`), built only with `-DPULSATRIX_ENABLE_VIZ=ON`:

```bash
cmake -S . -B build -DPULSATRIX_ENABLE_VIZ=ON
cmake --build build --target training_dashboard_demo --config Release
```

## Code

```cpp
ImPlotMetricsSink sink;
VizWindow window("Training Dashboard", 1000, 700);
window.run([&]() {
    for (int i = 0; i < 4; ++i) {
        net.train_step(input, target, optimizer, sink, step);  // logs loss via sink
    }
    ImGui::Begin("Training Dashboard");
    TrainingDashboard::Draw(sink);
    ImGui::End();
});
```

Full source: [`examples/viz/training_dashboard_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/viz/training_dashboard_demo.cpp).

## What you'll see

A window titled "pulsatrix -- Training Dashboard Demo" opens and immediately starts training
`XorNetwork` (Linear(2,4) -> ReLU -> Linear(4,1), Adam), a few steps per frame so the loss
curve animates instead of finishing before the first frame renders. The dashboard shows a
data-ink-minimal per-tag summary line (last/min/max value, current step) above a live line
chart of the loss.

## What's happening

`ImPlotMetricsSink` is a concrete `MetricsSink` — the same abstract logging interface every
training loop in this codebase already accepts, so wiring in live plotting needs zero changes
to `XorNetwork::train_step()` itself. `log_scalar()` buffers each step's loss into a per-tag
time series (unconditional, unit-tested even without `PULSATRIX_ENABLE_VIZ`); only
`TrainingDashboard::Draw()` — called once per frame inside `VizWindow::run()`'s callback —
touches ImPlot, reading the buffered series back out to draw the chart.

See also: [Visualization](../../visualization/index.md#a-live-training-dashboard).
