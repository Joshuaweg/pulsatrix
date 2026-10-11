// INT-2: linear CKA, checked against Kornblith et al.'s reference implementation
// (tools/generate_cka_reference_values.py), its invariances, and the roadmap's falsifier: CKA must
// tell a trained network from an untrained one of the same architecture.
#include "pulsatrix/representation_similarity.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

class CkaTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The reference script's inputs: x[i][a] = sin(seed + 0.37 i + 1.13 a + 0.05 i a), float32.
    Tensor Make(double seed, int64_t n, int64_t p) {
        std::vector<float> v;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t a = 0; a < p; ++a) {
                v.push_back(static_cast<float>(std::sin(seed + 0.37 * i + 1.13 * a + 0.05 * i * a)));
            }
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    Tensor Gaussian(int64_t n, int64_t p, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> v(static_cast<std::size_t>(n * p));
        for (float& x : v) {
            x = normal(rng);
        }
        return Tensor(Shape({n, p}), &backend, v);
    }

    // x times a p × p rotation (a product of Givens rotations), times scale, plus offset.
    Tensor RotateScaleShift(const Tensor& x, float scale, float offset) {
        const int64_t n = x.shape().dim(0);
        const int64_t p = x.shape().dim(1);
        const std::vector<float> host = x.to_host_vector();
        std::vector<double> v(host.begin(), host.end());
        for (int64_t a = 0; a + 1 < p; ++a) {
            const double t = 0.3 + 0.5 * static_cast<double>(a);
            for (int64_t i = 0; i < n; ++i) {
                double& u = v[static_cast<std::size_t>(i * p + a)];
                double& w = v[static_cast<std::size_t>(i * p + a + 1)];
                const double nu = std::cos(t) * u - std::sin(t) * w;
                const double nw = std::sin(t) * u + std::cos(t) * w;
                u = nu;
                w = nw;
            }
        }
        std::vector<float> out;
        for (double d : v) {
            out.push_back(static_cast<float>(d * scale + offset));
        }
        return Tensor(x.shape(), &backend, out);
    }
};

TEST_F(CkaTest, MatchesKornblithsReferenceImplementation) {
    struct Case {
        double seed_x, seed_y;
        int64_t n, p, q;
        double biased, debiased;
    };
    // More inputs than features (computed through the features), more features than inputs and a
    // square case (through Gram matrices).
    const Case cases[] = {
        {0.0, 2.0, 10, 4, 3, 0.94235544328613707, 0.93728990572982218},
        {0.5, 1.5, 6, 20, 15, 0.97283143698780472, 0.96490303485374418},
        {1.0, 3.0, 12, 12, 12, 0.99119835037167159, 0.98870342774364262},
    };
    for (const Case& c : cases) {
        const Tensor x = Make(c.seed_x, c.n, c.p);
        const Tensor y = Make(c.seed_y, c.n, c.q);
        EXPECT_NEAR(LinearCka(x, y, CkaEstimator::Biased), c.biased, 1e-12) << c.n << "x" << c.p;
        EXPECT_NEAR(LinearCka(x, y, CkaEstimator::Debiased), c.debiased, 1e-12) << c.n << "x" << c.p;
        EXPECT_NEAR(LinearCka(x, y), c.debiased, 1e-12);  // the default
    }
}

TEST_F(CkaTest, MatchesTheReferenceForANonlinearFunctionOfTheInput) {
    // y = tanh(x W), each rounded to float32 as in the script.
    const Tensor x = Make(0.25, 16, 6);
    const std::vector<float> xv = x.to_host_vector();
    const std::vector<float> wv = Make(4.0, 6, 5).to_host_vector();
    std::vector<float> yv;
    for (int64_t i = 0; i < 16; ++i) {
        for (int64_t c = 0; c < 5; ++c) {
            double s = 0;
            for (int64_t a = 0; a < 6; ++a) {
                s += static_cast<double>(xv[static_cast<std::size_t>(i * 6 + a)]) * wv[static_cast<std::size_t>(a * 5 + c)];
            }
            yv.push_back(static_cast<float>(std::tanh(s)));
        }
    }
    const Tensor y(Shape({16, 5}), &backend, yv);
    EXPECT_NEAR(LinearCka(x, y, CkaEstimator::Biased), 0.26465600145603679, 1e-9);
    EXPECT_NEAR(LinearCka(x, y, CkaEstimator::Debiased), 0.19573432900235432, 1e-9);
}

TEST_F(CkaTest, IsOneForRotatedScaledAndShiftedCopies) {
    for (const auto& shape : {std::vector<int64_t>{40, 6}, std::vector<int64_t>{8, 30}}) {
        const Tensor x = Gaussian(shape[0], shape[1], 3);
        const Tensor y = RotateScaleShift(x, 2.5f, -7.0f);
        for (CkaEstimator e : {CkaEstimator::Biased, CkaEstimator::Debiased}) {
            EXPECT_NEAR(LinearCka(x, x, e), 1.0, 1e-12);
            EXPECT_NEAR(LinearCka(x, y, e), 1.0, 1e-6) << shape[0] << "x" << shape[1];
        }
    }
}

TEST_F(CkaTest, IsSymmetricAndNotInvariantToStretchingOneFeature) {
    const Tensor x = Gaussian(30, 5, 4);
    const Tensor y = Gaussian(30, 7, 5);
    EXPECT_DOUBLE_EQ(LinearCka(x, y), LinearCka(y, x));
    // CKA is invariant to rotation and uniform scale only: stretching one feature changes it.
    std::vector<float> v = x.to_host_vector();
    for (std::size_t i = 0; i < v.size(); i += 5) {
        v[i] *= 10.0f;
    }
    EXPECT_LT(LinearCka(x, Tensor(x.shape(), &backend, v)), 0.9);
}

TEST_F(CkaTest, DebiasingRemovesTheBiasTowardsOneForFewWideInputs) {
    // Two unrelated random representations: 50 inputs of 1,000 features each.
    const Tensor x = Gaussian(50, 1000, 6);
    const Tensor y = Gaussian(50, 1000, 7);
    const double biased = LinearCka(x, y, CkaEstimator::Biased);
    const double debiased = LinearCka(x, y, CkaEstimator::Debiased);
    std::printf("CKA of unrelated 50 x 1000: biased %.4f, debiased %.4f\n", biased, debiased);
    EXPECT_GT(biased, 0.9);
    EXPECT_LT(std::fabs(debiased), 0.05);
}

TEST_F(CkaTest, FlattensDimensionsAfterTheFirst) {
    const Tensor flat = Gaussian(9, 24, 8);
    const Tensor image(Shape({9, 2, 3, 4}), &backend, flat.to_host_vector());
    const Tensor y = Gaussian(9, 5, 9);
    EXPECT_DOUBLE_EQ(LinearCka(flat, y), LinearCka(image, y));
}

TEST_F(CkaTest, IsNaNForAConstantRepresentation) {
    const Tensor x = Gaussian(10, 3, 10);
    const Tensor constant(Shape({10, 3}), &backend, std::vector<float>(30, 2.0f));
    EXPECT_TRUE(std::isnan(LinearCka(x, constant)));
    EXPECT_TRUE(std::isnan(LinearCka(constant, x, CkaEstimator::Biased)));
}

TEST_F(CkaTest, RefusesMismatchedOrTooSmallInputs) {
    const Tensor x = Gaussian(10, 3, 11);
    EXPECT_THROW((void)LinearCka(Tensor(Shape({10}), &backend, std::vector<float>(10, 1.0f)), x),
                 std::invalid_argument);
    EXPECT_THROW((void)LinearCka(x, Gaussian(9, 3, 12)), std::invalid_argument);
    EXPECT_THROW((void)LinearCka(Gaussian(3, 2, 13), Gaussian(3, 2, 14)), std::invalid_argument);
    EXPECT_NO_THROW((void)LinearCka(Gaussian(3, 2, 13), Gaussian(3, 2, 14), CkaEstimator::Biased));
    EXPECT_THROW((void)LinearCka(Gaussian(1, 2, 13), Gaussian(1, 2, 14), CkaEstimator::Biased),
                 std::invalid_argument);
}

TEST_F(CkaTest, MatrixComparesEveryPairBothWays) {
    const std::vector<Tensor> a = {Gaussian(20, 8, 20), Gaussian(20, 30, 21)};
    const std::vector<Tensor> b = {Gaussian(20, 5, 22), a[0], Gaussian(20, 40, 23)};
    for (CkaEstimator e : {CkaEstimator::Biased, CkaEstimator::Debiased}) {
        const CkaMatrix shared = LinearCkaMatrix(a, b, e);              // Gram matrices reused
        const CkaMatrix pairwise = LinearCkaMatrix(a, b, e, /*max_gram_bytes=*/0);  // pair by pair
        ASSERT_EQ(shared.rows, 2);
        ASSERT_EQ(shared.cols, 3);
        ASSERT_EQ(shared.values.size(), 6u);
        for (int64_t i = 0; i < 2; ++i) {
            for (int64_t j = 0; j < 3; ++j) {
                const double direct = LinearCka(a[static_cast<std::size_t>(i)], b[static_cast<std::size_t>(j)], e);
                EXPECT_NEAR(shared.at(i, j), direct, 1e-12);
                EXPECT_NEAR(pairwise.at(i, j), direct, 1e-12);
            }
        }
        EXPECT_NEAR(shared.at(0, 1), 1.0, 1e-12);
    }
    const CkaMatrix m = LinearCkaMatrix(a, b);
    EXPECT_THROW((void)m.at(2, 0), std::out_of_range);
    EXPECT_THROW((void)m.at(0, -1), std::out_of_range);
    EXPECT_THROW((void)LinearCkaMatrix({}, b), std::invalid_argument);
    EXPECT_THROW((void)LinearCkaMatrix(a, {Gaussian(19, 3, 24)}), std::invalid_argument);
}

// ---- the falsifier -----------------------------------------------------------------------------

// A 20 → 64 → 64 → 1 ReLU network. The target depends on two of the 20 inputs, so a trained
// network's hidden layers come to represent those two, and an untrained one's all twenty.
class SmallNetwork {
public:
    SmallNetwork(DeviceBackend* backend, uint32_t seed)
        : l0_(20, 64, backend), r0_(backend), l1_(64, 64, backend), r1_(backend), l2_(64, 1, backend),
          model_({&l0_, &r0_, &l1_, &r1_, &l2_}) {
        std::mt19937 rng(seed);
        for (LinearModule* l : {&l0_, &l1_, &l2_}) {
            const int64_t in = l->weight().shape().dim(0);
            const int64_t out = l->weight().shape().dim(1);
            std::uniform_real_distribution<float> u(-1.0f / std::sqrt(static_cast<float>(in)),
                                                    1.0f / std::sqrt(static_cast<float>(in)));
            std::vector<float> w(static_cast<std::size_t>(in * out));
            for (float& x : w) {
                x = u(rng);
            }
            l->set_weight(w);
        }
    }

    void Train(DeviceBackend* backend, const Tensor& x, const Tensor& y, int steps) {
        AdamOptimizer optimizer(0.01f, backend);
        MSELoss loss(backend);
        for (int s = 0; s < steps; ++s) {
            optimizer.zero_grad(model_);
            (void)loss.forward(model_.forward(x), y);
            (void)model_.backward(loss.backward());
            optimizer.step(model_);
        }
    }

    // The second hidden layer, after its ReLU.
    Tensor Hidden(const Tensor& x) { return r1_.forward(l1_.forward(r0_.forward(l0_.forward(x)))); }

private:
    LinearModule l0_;
    ReluModule r0_;
    LinearModule l1_;
    ReluModule r1_;
    LinearModule l2_;
    SequentialModule model_;
};

TEST_F(CkaTest, TellsATrainedNetworkFromAnUntrainedOne) {
    const Tensor x = Gaussian(256, 20, 30);
    std::vector<float> yv;
    const std::vector<float> xv = x.to_host_vector();
    for (int64_t i = 0; i < 256; ++i) {
        const float a = xv[static_cast<std::size_t>(i * 20)];
        const float b = xv[static_cast<std::size_t>(i * 20 + 1)];
        yv.push_back(std::sin(2.0f * a) + b * b);
    }
    const Tensor y(Shape({256, 1}), &backend, yv);

    SmallNetwork trained_a(&backend, 1);
    SmallNetwork trained_b(&backend, 2);
    SmallNetwork untrained(&backend, 3);
    trained_a.Train(&backend, x, y, 150);
    trained_b.Train(&backend, x, y, 150);

    const Tensor probe = Gaussian(200, 20, 31);
    const Tensor ha = trained_a.Hidden(probe);
    const Tensor hb = trained_b.Hidden(probe);
    const Tensor hu = untrained.Hidden(probe);
    const double trained_pair = LinearCka(ha, hb);
    const double against_untrained = LinearCka(ha, hu);
    const double untrained_inputs = LinearCka(probe, hu);
    // Measured: 0.68 between the trained networks, 0.10 against the untrained one, which
    // represents its random input far more than either trained network does.
    EXPECT_GT(trained_pair, against_untrained + 0.3) << trained_pair << " vs " << against_untrained;
    EXPECT_GT(untrained_inputs, against_untrained) << untrained_inputs;
    std::printf("CKA: trained-trained %.4f, trained-untrained %.4f, untrained-input %.4f\n", trained_pair,
                against_untrained, untrained_inputs);
}

}  // namespace
}  // namespace pulsatrix
