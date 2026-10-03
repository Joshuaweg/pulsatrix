# Recipe: Live Training Dashboard

**What you'll build:** a GUI window showing `XorNetwork` training in real time. The loss curve
animates as `ImPlotMetricsSink` logs each step and `TrainingDashboard` draws the result.

This is a GUI demo, not a console recipe. It needs `-DPULSATRIX_ENABLE_VIZ=ON` and a display.

CMake target: `training_dashboard_demo`
(`examples/viz/training_dashboard_demo.cpp`):

```bash
cmake -S . -B build -DPULSATRIX_ENABLE_VIZ=ON
cmake --build build --target training_dashboard_demo --config Release
```

Run it: `./build/training_dashboard_demo` (Windows: `build\Release\training_dashboard_demo.exe`).

## Code

```cpp
ImPlotMetricsSink sink;
VizWindow window("pulsatrix -- Training Dashboard Demo", 1000, 700);
window.run([&]() {
    for (int i = 0; i < 4 && step < kMaxSteps; ++i, ++step) {
        // ... build input and target_tensor for XOR example (step % 4) ...
        net.train_step(input, target_tensor, optimizer, sink, step);  // logs loss via sink
    }
    ImGui::Begin("Training Dashboard");
    TrainingDashboard::Draw(sink);
    ImGui::End();
});
```

Full source: [`examples/viz/training_dashboard_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/viz/training_dashboard_demo.cpp).

## What you'll see

A window titled "pulsatrix -- Training Dashboard Demo" opens and starts training `XorNetwork`
(Linear(2,4) -> ReLU -> Linear(4,1), Adam) right away. It runs a few steps per frame, so the
loss curve animates instead of finishing before the first frame renders. The dashboard shows a
compact summary line per metric (last, min and max value, current step) above a live line chart
of the loss.

## What's happening

`ImPlotMetricsSink` is a `MetricsSink`, the same logging interface every training loop in this
codebase accepts. Live plotting therefore needs no changes to `XorNetwork::train_step()`.

`log_scalar()` buffers each step's loss into a time series per metric name. That part always
builds and is unit-tested even without `PULSATRIX_ENABLE_VIZ`. Only `TrainingDashboard::Draw()`
touches ImPlot. It runs once per frame inside `VizWindow::run()`'s callback and draws the
buffered series.

See also: [Visualization](../../visualization/index.md#a-live-training-dashboard).
