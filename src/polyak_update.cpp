#include "exai/polyak_update.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "exai/shape.hpp"
#include "exai/tensor.hpp"

namespace exai {
namespace {

// Renders a Shape as "(a, b, ...)" for an error message. Deliberately duplicated from
// dqn_target.cpp's own TU-local helper rather than promoted to a shared header: promoting it
// would mean editing dqn_target.cpp, which this mission explicitly does not touch
// (SyncTargetNetwork is not modified). Two five-line formatters are cheaper than a speculative
// "shape formatting" utility header; if a third caller appears, that is the moment to hoist it.
std::string shape_to_string(const Shape& shape) {
    std::string out = "(";
    for (int64_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape.dim(static_cast<size_t>(i)));
    }
    return out + ")";
}

}  // namespace

void PolyakUpdate(Module& source, Module& destination, float tau) {
    // Checked first, before any buffer is touched: a bad tau invalidates every element, so
    // there is no sense in blending some of them before finding out.
    if (!(tau > 0.0f && tau <= 1.0f)) {
        throw std::invalid_argument("PolyakUpdate: tau must be in (0, 1] -- tau == 0 would leave the destination "
                                    "permanently frozen, which is a caller error, not a no-op");
    }

    std::vector<ParamRef> source_params = source.parameters();
    std::vector<ParamRef> destination_params = destination.parameters();

    if (source_params.size() != destination_params.size()) {
        throw std::invalid_argument("PolyakUpdate: source exposes " + std::to_string(source_params.size()) +
                                    " parameters but destination exposes " +
                                    std::to_string(destination_params.size()) +
                                    " -- the two networks have different architectures");
    }

    for (size_t i = 0; i < source_params.size(); ++i) {
        const Tensor& from = *source_params[i].value;
        Tensor& into = *destination_params[i].value;
        if (from.shape() != into.shape()) {
            throw std::invalid_argument("PolyakUpdate: parameter " + std::to_string(i) + " has shape " +
                                        shape_to_string(from.shape()) + " in source but " +
                                        shape_to_string(into.shape()) + " in destination");
        }
        // Element-wise into the *existing* buffer, never `into = from`: destination's parameter
        // Tensors must remain the same objects its own parameters() -- and any optimizer
        // already holding ParamRefs into them -- point at. The destination's current value is
        // also a read operand here, which is what makes successive calls an exponential moving
        // average rather than a sequence of independent writes.
        for (int64_t e = 0; e < from.numel(); ++e) {
            into.data()[e] = tau * from.data()[e] + (1.0f - tau) * into.data()[e];
        }
    }
}

}  // namespace exai
