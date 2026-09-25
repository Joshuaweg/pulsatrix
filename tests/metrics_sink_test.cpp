#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/metrics_sink.hpp"

namespace pulsatrix {
namespace {

TEST(NoOpMetricsSinkTest, LogScalarDoesNotThrow) {
    NoOpMetricsSink sink;
    EXPECT_NO_THROW(sink.log_scalar("loss", 1.5, 0));
}

TEST(NoOpMetricsSinkTest, LogHistogramDoesNotThrow) {
    CPUBackend backend;
    NoOpMetricsSink sink;
    Tensor values(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    EXPECT_NO_THROW(sink.log_histogram("weights", values, 0));
}

TEST(NoOpMetricsSinkTest, CallableThroughBasePointer) {
    NoOpMetricsSink sink;
    MetricsSink& base = sink;
    EXPECT_NO_THROW(base.log_scalar("loss", 0.1, 5));
}

}  // namespace
}  // namespace pulsatrix
