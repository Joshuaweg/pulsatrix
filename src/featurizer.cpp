#include "pulsatrix/featurizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulsatrix {

FeaturizerLoss Featurizer::loss_and_backward_with_target(const Tensor&, const Tensor&, std::vector<float>*) {
    throw std::invalid_argument("Featurizer::loss_and_backward_with_target: this featurizer reconstructs its input; use loss_and_backward()");
}

FeatureActivityTracker::FeatureActivityTracker(int64_t num_features) {
    if (num_features < 1) throw std::invalid_argument("FeatureActivityTracker: num_features must be at least 1");
    fired_.assign(static_cast<size_t>(num_features), 0);
    last_fired_.assign(static_cast<size_t>(num_features), -1);
}

void FeatureActivityTracker::observe(const std::vector<float>& codes, float threshold) {
    const size_t m = fired_.size();
    if (codes.size() % m != 0) throw std::invalid_argument("FeatureActivityTracker::observe: codes must be whole rows of num_features");
    for (size_t row = 0; row < codes.size() / m; ++row) {
        for (size_t i = 0; i < m; ++i) {
            if (std::abs(codes[row * m + i]) > threshold) {
                ++fired_[i];
                last_fired_[i] = inputs_;
            }
        }
        ++inputs_;
    }
}

std::vector<double> FeatureActivityTracker::firing_rates() const {
    std::vector<double> out(fired_.size(), 0.0);
    if (inputs_ == 0) return out;
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(fired_[i]) / static_cast<double>(inputs_);
    return out;
}

std::vector<int64_t> FeatureActivityTracker::dead(int64_t window) const {
    if (window < 1) throw std::invalid_argument("FeatureActivityTracker::dead: window must be at least 1");
    std::vector<int64_t> out;
    for (size_t i = 0; i < last_fired_.size(); ++i) {
        // Fired within the last `window` inputs: last_fired >= inputs - window.
        if (last_fired_[i] < 0 || last_fired_[i] < inputs_ - window) out.push_back(static_cast<int64_t>(i));
    }
    return out;
}

double FeatureActivityTracker::dead_fraction(int64_t window) const {
    return static_cast<double>(dead(window).size()) / static_cast<double>(fired_.size());
}

std::vector<int64_t> FeatureActivityTracker::dense(double rate) const {
    std::vector<int64_t> out;
    const std::vector<double> rates = firing_rates();
    for (size_t i = 0; i < rates.size(); ++i) {
        if (rates[i] > rate) out.push_back(static_cast<int64_t>(i));
    }
    return out;
}

void FeatureActivityTracker::reset() {
    inputs_ = 0;
    std::fill(fired_.begin(), fired_.end(), 0);
    std::fill(last_fired_.begin(), last_fired_.end(), -1);
}

double MeanL0(const std::vector<float>& codes, int64_t num_features, float threshold) {
    if (num_features < 1 || codes.empty() || codes.size() % static_cast<size_t>(num_features) != 0) {
        throw std::invalid_argument("MeanL0: codes must be one or more whole rows of num_features");
    }
    int64_t active = 0;
    for (float c : codes) active += std::abs(c) > threshold ? 1 : 0;
    return static_cast<double>(active) / static_cast<double>(codes.size() / static_cast<size_t>(num_features));
}

double MeanL0(Featurizer& featurizer, const Tensor& x, float threshold) {
    const std::vector<float> codes = featurizer.encode(x).to_host_vector();
    const int64_t b = featurizer.block_size();
    if (b == 1) return MeanL0(codes, featurizer.num_features(), threshold);
    int64_t active = 0;
    for (size_t g = 0; g < codes.size(); g += static_cast<size_t>(b)) {
        bool on = false;
        for (int64_t s = 0; s < b; ++s) on = on || std::abs(codes[g + static_cast<size_t>(s)]) > threshold;
        active += on ? 1 : 0;
    }
    return static_cast<double>(active) / static_cast<double>(x.shape().dim(0));
}

}  // namespace pulsatrix
