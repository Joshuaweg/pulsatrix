// KS-4 on the GPU: PGD through a network on the HIP backend gives the CPU's adversarial inputs.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/adversarial.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

AttackResult Attack(DeviceBackend* backend, AttackNorm norm) {
    LinearModule l1(4, 8, backend), l2(8, 3, backend);
    ReluModule relu(backend);
    SequentialModule net({&l1, &relu, &l2});
    std::vector<float> w1, w2, x;
    for (int k = 0; k < 32; ++k) w1.push_back(static_cast<float>(0.7 * std::sin(1.3 * k + 0.2)));
    for (int k = 0; k < 24; ++k) w2.push_back(static_cast<float>(0.6 * std::cos(0.9 * k + 0.4)));
    for (int k = 0; k < 24; ++k) x.push_back(static_cast<float>(0.5 + 0.4 * std::sin(0.8 * k + 0.3)));
    l1.set_weight(w1);
    l2.set_weight(w2);
    net.set_training(false);
    AttackOptions o;
    o.epsilon = norm == AttackNorm::Linf ? 0.1 : 0.4;
    o.norm = norm;
    o.steps = 8;
    o.seed = 3;
    o.clip_min = 0;
    o.clip_max = 1;
    return Pgd(ClassifierFrom(net), Tensor(Shape({6, 4}), backend, x, backend->device()), {0, 1, 2, 0, 1, 2}, backend, o);
}

TEST(AdversarialHip, PgdOnTheGpuGivesTheCpusAttack) {
    HIPBackend hip;
    CPUBackend cpu;
    for (AttackNorm norm : {AttackNorm::Linf, AttackNorm::L2}) {
        const AttackResult g = Attack(&hip, norm), c = Attack(&cpu, norm);
        EXPECT_EQ(g.adversarial.device(), DeviceType::Hip);
        const std::vector<float> a = g.adversarial.to_host_vector(), b = c.adversarial.to_host_vector();
        for (size_t k = 0; k < a.size(); ++k) EXPECT_NEAR(a[k], b[k], 1e-5) << k;
        EXPECT_EQ(g.adversarial_predicted, c.adversarial_predicted);
        EXPECT_EQ(g.success, c.success);
    }
}

}  // namespace
}  // namespace pulsatrix
