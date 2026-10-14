// KD-4: born-again networks' ensemble and permuted-dark-knowledge control, against
// tools/generate_layer_student_reference_values.py (numpy).
#include "pulsatrix/born_again.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

namespace reference {
const std::vector<double> kEnsemble = {-2.212555743368, -0.810773029092, -1.214376408266, -1.902669009395, -0.940086293544, -1.571139812678, -1.191840051880, -2.323444229201, -1.567752290921, -1.615152115688, -1.078060343591, -1.376853527206, -1.368489518962, -2.371499079345, -2.185826787282, -0.616587231862};
const std::vector<double> kPermuted = {0.794677317142, 0.956997334957, 3.637189626694, -2.447431564331, 3.564829349518, -2.525066614151, 3.594832420349, 0.860479950905, -3.999692916870, 3.608687400818, -2.500282526016, -2.472548484802, -3.999960899353, 0.891559660435, 0.926039278507};
}  // namespace reference

class BornAgainTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The generator's inputs: scale * float32(sin(a k + b)) of the flat index k.
    Tensor Logits(const Shape& shape, double a, double b, float scale) {
        std::vector<float> v(static_cast<size_t>(shape.numel()));
        for (size_t k = 0; k < v.size(); ++k) v[k] = scale * static_cast<float>(std::sin(a * static_cast<double>(k) + b));
        return Tensor(shape, &backend, v);
    }
};

TEST_F(BornAgainTest, TheEnsembleIsTheLogOfTheMeanDistribution) {
    const Shape shape({2, 2, 4});
    const Tensor out = EnsembleLogProbabilities({Logits(shape, 0.8, 0.1, 3), Logits(shape, 0.3, 0.9, 3), Logits(shape, 1.3, 0.4, 3)}, &backend);
    EXPECT_EQ(out.shape(), shape);
    const std::vector<float> got = out.to_host_vector();
    for (size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], reference::kEnsemble[k], 1e-6) << k;
}

TEST_F(BornAgainTest, OneGenerationsEnsembleIsItsLogSoftmax) {
    const Tensor l = Logits(Shape({3, 5}), 0.6, 0.2, 50);  // large logits: the shift keeps it finite
    const std::vector<float> got = EnsembleLogProbabilities({l}, &backend).to_host_vector();
    const std::vector<float> v = l.to_host_vector();
    for (size_t r = 0; r < 3; ++r) {
        const double top = *std::max_element(v.begin() + static_cast<std::ptrdiff_t>(5 * r), v.begin() + static_cast<std::ptrdiff_t>(5 * r + 5));
        double sum = 0;
        for (size_t k = 0; k < 5; ++k) sum += std::exp(v[5 * r + k] - top);
        for (size_t k = 0; k < 5; ++k) EXPECT_NEAR(got[5 * r + k], v[5 * r + k] - top - std::log(sum), 1e-4);
    }
}

TEST_F(BornAgainTest, TheEnsembleRefusesBadInput) {
    EXPECT_THROW((void)EnsembleLogProbabilities({}, &backend), std::invalid_argument);
    EXPECT_THROW((void)EnsembleLogProbabilities({Logits(Shape({4}), 0.1, 0, 1)}, &backend), std::invalid_argument);
    EXPECT_THROW((void)EnsembleLogProbabilities({Logits(Shape({2, 4}), 0.1, 0, 1), Logits(Shape({4, 2}), 0.1, 0, 1)}, &backend),
                 std::invalid_argument);
}

TEST_F(BornAgainTest, ThePermutationMatchesTheReference) {
    const std::vector<float> got = PermuteDarkKnowledge(Logits(Shape({3, 5}), 0.9, 0.2, 4), 11, &backend).to_host_vector();
    ASSERT_EQ(got.size(), reference::kPermuted.size());
    for (size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], reference::kPermuted[k], 1e-6) << k;
}

TEST_F(BornAgainTest, ThePermutationKeepsTheTopClassAndEachRowsValues) {
    const Tensor l = Logits(Shape({6, 2, 9}), 0.37, 0.5, 3);
    const std::vector<float> v = l.to_host_vector();
    const std::vector<float> a = PermuteDarkKnowledge(l, 1, &backend).to_host_vector();
    const std::vector<float> b = PermuteDarkKnowledge(l, 2, &backend).to_host_vector();
    EXPECT_EQ(a, PermuteDarkKnowledge(l, 1, &backend).to_host_vector());
    EXPECT_NE(a, b);
    EXPECT_NE(a, v);
    for (size_t r = 0; r < 12; ++r) {
        const auto row = [&](const std::vector<float>& x) { return std::vector<float>(x.begin() + static_cast<std::ptrdiff_t>(9 * r), x.begin() + static_cast<std::ptrdiff_t>(9 * r + 9)); };
        const std::vector<float> before = row(v), after = row(a);
        const auto top = std::max_element(before.begin(), before.end()) - before.begin();
        EXPECT_EQ(after[static_cast<size_t>(top)], before[static_cast<size_t>(top)]) << r;
        std::vector<float> x = before, y = after;
        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());
        EXPECT_EQ(x, y) << r;
    }
    EXPECT_THROW((void)PermuteDarkKnowledge(Logits(Shape({9}), 0.1, 0, 1), 1, &backend), std::invalid_argument);
}

TEST_F(BornAgainTest, ATieKeepsTheFirstLargestInPlace) {
    const Tensor l(Shape({1, 4}), &backend, {2.0f, 5.0f, 5.0f, 1.0f});
    for (uint64_t seed = 0; seed < 20; ++seed) EXPECT_EQ(PermuteDarkKnowledge(l, seed, &backend).to_host_vector()[1], 5.0f);
}

}  // namespace
}  // namespace pulsatrix
