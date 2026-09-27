#include "pulsatrix/text_collate.hpp"

#include <algorithm>
#include <stdexcept>

namespace pulsatrix {

namespace {
Batch PadCollateImpl(std::vector<Sample> samples, DeviceBackend* backend, float pad_index) {
    if (samples.empty()) {
        throw std::invalid_argument("PadCollate: samples must not be empty");
    }

    int64_t max_len = 0;
    for (const Sample& sample : samples) {
        max_len = std::max(max_len, sample.fields[0].shape().dim(1));
    }
    int64_t n = static_cast<int64_t>(samples.size());

    Tensor padded(Shape({n, max_len}), backend);
    Tensor lengths(Shape({n}), backend);
    padded.fill(pad_index);

    for (int64_t i = 0; i < n; ++i) {
        const Tensor& sequence = samples[static_cast<size_t>(i)].fields[0];
        int64_t len = sequence.shape().dim(1);
        lengths.at({i}) = static_cast<float>(len);
        for (int64_t j = 0; j < len; ++j) {
            padded.at({i, j}) = sequence.at({0, j});
        }
    }

    Batch batch;
    batch.fields.push_back(std::move(padded));
    batch.fields.push_back(std::move(lengths));
    return batch;
}
}  // namespace

CollateFn PadCollate(float pad_index) {
    return [pad_index](std::vector<Sample> samples, DeviceBackend* backend) {
        return PadCollateImpl(std::move(samples), backend, pad_index);
    };
}

}  // namespace pulsatrix
