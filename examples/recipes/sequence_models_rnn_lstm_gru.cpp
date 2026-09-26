/** @file sequence_models_rnn_lstm_gru.cpp
 *  @brief Recipe: trains RNNModule, LSTMModule, and GRUModule on the same running-parity
 *         (cumulative XOR) task and reports each one's final accuracy. Paired with
 *         docs/recipes/deep-learning/sequence_models_rnn_lstm_gru.md.
 *  @note Uses the exact same dataset, seeds, and hyperparameters as
 *        examples/sequence_model_demo.cpp, just with terser per-epoch output -- see that
 *        file for the full per-sequence breakdown and design rationale.
 */
#include <cstdio>
#include <random>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/gru_module.hpp"
#include "pulsatrix/lstm_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/rnn_module.hpp"

namespace {

constexpr int kNumSequences = 6;
constexpr int kSeqLength = 10;
constexpr int kEpochs = 2000;
constexpr unsigned kSeed = 20260922u;

struct ParityDataset {
    std::vector<float> inputs;
    std::vector<float> targets;
};

ParityDataset make_dataset() {
    ParityDataset data;
    std::mt19937 rng(kSeed);
    std::bernoulli_distribution bit(0.5);
    for (int n = 0; n < kNumSequences; ++n) {
        int parity = 0;
        for (int t = 0; t < kSeqLength; ++t) {
            const int b = bit(rng) ? 1 : 0;
            parity ^= b;
            data.inputs.push_back(static_cast<float>(b));
            data.targets.push_back(static_cast<float>(parity));
        }
    }
    return data;
}

class Initializer {
public:
    explicit Initializer(unsigned seed) : rng_(seed), dist_(-1.0f, 1.0f) {}
    float operator()() { return dist_(rng_); }

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
};

void train_and_report(const char* label, pulsatrix::Module& module, const pulsatrix::Tensor& input,
                       const pulsatrix::Tensor& target, pulsatrix::DeviceBackend* backend, float learning_rate) {
    pulsatrix::AdamOptimizer optimizer(learning_rate, backend);
    pulsatrix::MSELoss loss(backend);

    float final_loss = 0.0f;
    for (int epoch = 0; epoch <= kEpochs; ++epoch) {
        optimizer.zero_grad(module);
        pulsatrix::Tensor prediction = module.forward(input);
        final_loss = loss.forward(prediction, target);
        if (epoch == kEpochs) {
            break;
        }
        pulsatrix::Tensor grad = loss.backward();
        (void)module.backward(grad);
        optimizer.step(module);
    }

    pulsatrix::Tensor prediction = module.forward(input);
    int correct = 0;
    const int total = kNumSequences * kSeqLength;
    for (int64_t i = 0; i < total; ++i) {
        const int rounded = (prediction.data()[i] > 0.5f) ? 1 : 0;
        const int expected = static_cast<int>(target.data()[i] + 0.5f);
        correct += (rounded == expected) ? 1 : 0;
    }
    std::printf("%-10s | final loss %.6f | timestep accuracy %3d/%d (%.1f%%)\n", label, final_loss, correct, total,
                100.0f * static_cast<float>(correct) / static_cast<float>(total));
}

}  // namespace

int main() {
    using namespace pulsatrix;

    CPUBackend backend;
    const ParityDataset data = make_dataset();
    const Shape shape({kNumSequences, kSeqLength, 1});
    const Tensor input(shape, &backend, data.inputs);
    const Tensor target(shape, &backend, data.targets);

    std::printf("Sequence-model recipe -- running parity of %d seeded sequences (length %d)\n\n", kNumSequences,
                kSeqLength);

    Initializer init(kSeed);

    RNNModule rnn(1, 1, &backend);
    rnn.set_weight_xh({init()});
    rnn.set_weight_hh({init()});
    rnn.set_bias({init()});

    LSTMModule lstm(1, 1, &backend);
    lstm.set_weight_xi({init()});
    lstm.set_weight_hi({init()});
    lstm.set_bias_i({init()});
    lstm.set_weight_xf({init()});
    lstm.set_weight_hf({init()});
    lstm.set_bias_f({init()});
    lstm.set_weight_xg({init()});
    lstm.set_weight_hg({init()});
    lstm.set_bias_g({init()});
    lstm.set_weight_xo({init()});
    lstm.set_weight_ho({init()});
    lstm.set_bias_o({init()});

    GRUModule gru(1, 1, &backend);
    gru.set_weight_xz({init()});
    gru.set_weight_hz({init()});
    gru.set_bias_z({init()});
    gru.set_weight_xr({init()});
    gru.set_weight_hr({init()});
    gru.set_bias_r({init()});
    gru.set_weight_xn({init()});
    gru.set_weight_hn({init()});
    gru.set_bias_n({init()});

    train_and_report("RNNModule", rnn, input, target, &backend, 0.05f);
    train_and_report("LSTMModule", lstm, input, target, &backend, 0.05f);
    train_and_report("GRUModule", gru, input, target, &backend, 0.05f);

    std::printf(
        "\nExpected: RNNModule plateaus (a single tanh unit cannot represent XOR(h,x) -- a\n"
        "representational limit, not an optimization failure); LSTM/GRU solve it exactly via\n"
        "their multiplicative gates. See examples/sequence_model_demo.cpp for the full proof.\n");

    return 0;
}
