#include "pulsatrix/tda/point_cloud.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

[[noreturn]] void Fail(const std::string& message) { throw std::invalid_argument("PointCloud: " + message); }

// A batch read as (samples, positions, features), with the feature axis moved last: element
// (s, p, f) is at ((s * positions) + p) * features + f.
struct Layout {
    int64_t samples = 0;
    int64_t positions = 1;
    int64_t features = 0;
    std::vector<float> values;
};

Layout ReadBatch(const Tensor& batch, const PointCloudOptions& options) {
    const int64_t rank = batch.rank();
    if (rank < 2) Fail("a batch must have shape (batch, ...), rank 2 or more");
    const Shape& shape = batch.shape();
    for (int64_t a = 0; a < rank; ++a) {
        if (shape.dim(static_cast<std::size_t>(a)) <= 0) Fail("a batch has an empty dimension");
    }
    std::vector<float> host = batch.to_host_vector();
    if (std::any_of(host.begin(), host.end(), [](float v) { return !std::isfinite(v); })) {
        Fail("a batch has a NaN or infinite value");
    }

    Layout out;
    out.samples = shape.dim(0);
    if (options.pooling == PointPooling::Flatten) {
        out.features = batch.numel() / out.samples;
        out.values = std::move(host);
        return out;
    }

    const int64_t axis = options.feature_axis < 0 ? options.feature_axis + rank : options.feature_axis;
    if (axis < 1 || axis >= rank) {
        Fail("feature_axis " + std::to_string(options.feature_axis) + " is not a feature axis of a rank-" +
             std::to_string(rank) + " batch (it can't be the batch axis)");
    }
    // Viewed as (samples, before, features, after): the positions are before x after.
    int64_t before = 1;
    for (int64_t a = 1; a < axis; ++a) before *= shape.dim(static_cast<std::size_t>(a));
    int64_t after = 1;
    for (int64_t a = axis + 1; a < rank; ++a) after *= shape.dim(static_cast<std::size_t>(a));
    out.features = shape.dim(static_cast<std::size_t>(axis));
    out.positions = before * after;
    if (after == 1) {
        out.values = std::move(host);  // the features are already last
        return out;
    }
    out.values.resize(host.size());
    const int64_t f_count = out.features;
    for (int64_t s = 0; s < out.samples; ++s) {
        for (int64_t b = 0; b < before; ++b) {
            for (int64_t f = 0; f < f_count; ++f) {
                const float* src = &host[static_cast<std::size_t>(((s * before + b) * f_count + f) * after)];
                for (int64_t t = 0; t < after; ++t) {
                    const int64_t position = b * after + t;
                    out.values[static_cast<std::size_t>((s * out.positions + position) * f_count + f)] = src[t];
                }
            }
        }
    }
    return out;
}

}  // namespace

PointCloud::PointCloud(PointCloudOptions options) : options_(options) {
    switch (options_.pooling) {
        case PointPooling::Flatten:
        case PointPooling::PerPosition:
        case PointPooling::Mean:
        case PointPooling::Max:
        case PointPooling::Last:
            break;
        default:
            Fail("unknown pooling");
    }
}

void PointCloud::add(const Tensor& batch) {
    const Layout in = ReadBatch(batch, options_);
    if (dimension_ != 0 && in.features != dimension_) {
        Fail("a batch gives points of dimension " + std::to_string(in.features) + ", but the cloud's are " +
             std::to_string(dimension_));
    }
    const std::size_t d = static_cast<std::size_t>(in.features);
    auto at = [&](int64_t s, int64_t p) { return &in.values[static_cast<std::size_t>(s * in.positions + p) * d]; };

    // Build the new points aside, so a failure above leaves the cloud as it was.
    std::vector<float> points;
    std::vector<PointOrigin> origins;
    for (int64_t s = 0; s < in.samples; ++s) {
        const int64_t sample = samples_ + s;
        switch (options_.pooling) {
            case PointPooling::Flatten:
                points.insert(points.end(), at(s, 0), at(s, 0) + d);
                origins.push_back({sample, -1});
                break;
            case PointPooling::PerPosition:
                for (int64_t p = 0; p < in.positions; ++p) {
                    points.insert(points.end(), at(s, p), at(s, p) + d);
                    origins.push_back({sample, p});
                }
                break;
            case PointPooling::Mean: {
                std::vector<double> sum(d, 0.0);
                for (int64_t p = 0; p < in.positions; ++p) {
                    for (std::size_t f = 0; f < d; ++f) sum[f] += at(s, p)[f];
                }
                for (double v : sum) points.push_back(static_cast<float>(v / static_cast<double>(in.positions)));
                origins.push_back({sample, -1});
                break;
            }
            case PointPooling::Max: {
                std::vector<float> best(at(s, 0), at(s, 0) + d);
                for (int64_t p = 1; p < in.positions; ++p) {
                    for (std::size_t f = 0; f < d; ++f) best[f] = std::max(best[f], at(s, p)[f]);
                }
                points.insert(points.end(), best.begin(), best.end());
                origins.push_back({sample, -1});
                break;
            }
            case PointPooling::Last:
                points.insert(points.end(), at(s, in.positions - 1), at(s, in.positions - 1) + d);
                origins.push_back({sample, in.positions - 1});
                break;
        }
    }
    dimension_ = in.features;
    samples_ += in.samples;
    values_.insert(values_.end(), points.begin(), points.end());
    origins_.insert(origins_.end(), origins.begin(), origins.end());
}

void PointCloud::add(const ActivationSnapshot& snapshot, NodeId node) {
    const std::vector<NodeId>& ids = snapshot.node_ids();
    if (std::find(ids.begin(), ids.end(), node) == ids.end()) {
        Fail("node " + std::to_string(node) + " is not in the snapshot");
    }
    add(snapshot.activation(node));
}

Tensor PointCloud::ToTensor(DeviceBackend* backend) const {
    if (origins_.empty()) Fail("the cloud is empty");
    return Tensor(Shape({size(), dimension_}), backend, values_);
}

PointCloud MakePointCloud(const std::vector<ActivationSnapshot>& snapshots, NodeId node, PointCloudOptions options) {
    if (snapshots.empty()) Fail("no snapshots");
    PointCloud cloud(options);
    for (const ActivationSnapshot& s : snapshots) cloud.add(s, node);
    return cloud;
}

}  // namespace pulsatrix
