#include "pulsatrix/collate.hpp"

#include <stdexcept>

namespace pulsatrix {

Batch DefaultCollate(std::vector<Sample> samples, DeviceBackend* backend) {
    if (samples.empty()) {
        throw std::invalid_argument("DefaultCollate: samples must not be empty");
    }

    size_t field_count = samples[0].fields.size();
    for (const Sample& sample : samples) {
        if (sample.fields.size() != field_count) {
            throw std::invalid_argument("DefaultCollate: every sample must have the same field count");
        }
    }

    Batch batch;
    batch.fields.reserve(field_count);
    for (size_t field_idx = 0; field_idx < field_count; ++field_idx) {
        std::vector<Tensor> field_tensors;
        field_tensors.reserve(samples.size());
        for (const Sample& sample : samples) {
            field_tensors.push_back(sample.fields[field_idx]);
        }
        batch.fields.push_back(Tensor::Stack(field_tensors, backend));
    }
    return batch;
}

}  // namespace pulsatrix
