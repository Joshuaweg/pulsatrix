// Born-again networks (KD-4): the generations' ensemble, and the permuted-dark-knowledge control.

#include "pulsatrix/born_again.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "portable_random.hpp"

namespace pulsatrix {

Tensor EnsembleLogProbabilities(const std::vector<Tensor>& logits, DeviceBackend* backend) {
    if (logits.empty()) throw std::invalid_argument("EnsembleLogProbabilities: no generations");
    const Tensor& first = logits.front();
    if (first.rank() < 2) throw std::invalid_argument("EnsembleLogProbabilities: the rank must be at least 2");
    for (const Tensor& l : logits) {
        if (l.shape() != first.shape()) throw std::invalid_argument("EnsembleLogProbabilities: the shapes differ");
        if (l.device() != first.device()) throw std::invalid_argument("EnsembleLogProbabilities: the tensors are on different devices");
    }
    const int64_t classes = first.shape().dim(first.rank() - 1);
    const int64_t rows = first.numel() / classes;
    std::vector<double> mean(static_cast<size_t>(first.numel()), 0.0);
    for (const Tensor& l : logits) {
        const std::vector<float> v = l.to_host_vector();
        for (int64_t r = 0; r < rows; ++r) {
            const size_t o = static_cast<size_t>(r * classes);
            const double top = *std::max_element(v.begin() + static_cast<std::ptrdiff_t>(o), v.begin() + static_cast<std::ptrdiff_t>(o + classes));
            double sum = 0.0;
            for (int64_t k = 0; k < classes; ++k) sum += std::exp(v[o + static_cast<size_t>(k)] - top);
            for (int64_t k = 0; k < classes; ++k) mean[o + static_cast<size_t>(k)] += std::exp(v[o + static_cast<size_t>(k)] - top) / sum;
        }
    }
    const auto generations = static_cast<double>(logits.size());
    std::vector<float> out(mean.size());
    for (size_t i = 0; i < mean.size(); ++i) out[i] = static_cast<float>(std::log(mean[i] / generations));
    return Tensor(first.shape(), backend, out, first.device());
}

Tensor PermuteDarkKnowledge(const Tensor& teacher_logits, uint64_t seed, DeviceBackend* backend) {
    if (teacher_logits.rank() < 2) throw std::invalid_argument("PermuteDarkKnowledge: the rank must be at least 2");
    const int64_t classes = teacher_logits.shape().dim(teacher_logits.rank() - 1);
    const int64_t rows = teacher_logits.numel() / classes;
    std::vector<float> v = teacher_logits.to_host_vector();
    PortableRng rng{seed};
    std::vector<size_t> others;
    for (int64_t r = 0; r < rows; ++r) {
        const size_t o = static_cast<size_t>(r * classes);
        const auto top = static_cast<size_t>(std::max_element(v.begin() + static_cast<std::ptrdiff_t>(o),
                                                              v.begin() + static_cast<std::ptrdiff_t>(o + classes)) -
                                             v.begin());
        others.clear();
        for (int64_t k = 0; k < classes; ++k) {
            if (o + static_cast<size_t>(k) != top) others.push_back(o + static_cast<size_t>(k));
        }
        // Fisher-Yates over the other classes' values.
        for (size_t i = others.size(); i > 1; --i) {
            const auto j = static_cast<size_t>(rng.next() % i);
            std::swap(v[others[i - 1]], v[others[j]]);
        }
    }
    return Tensor(teacher_logits.shape(), backend, v, teacher_logits.device());
}

}  // namespace pulsatrix
