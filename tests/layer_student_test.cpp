// KD-4: DistilBERT's student. A student with the teacher's layers computes what the teacher does;
// the cosine loss matches tools/generate_layer_student_reference_values.py (numpy, gradients by
// central differences); and the tap trains a model on a loss of its hidden states.
#include "pulsatrix/layer_student.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lora.hpp"

namespace pulsatrix {
namespace {

namespace reference {
const double kCosineLoss = 1.002828142033;
const std::vector<double> kCosineGrad = {0.013114971109, 0.018531140844, 0.004046280644, -0.030349172953, -0.071715087535, -0.053275950096, -0.003057209597, 0.039300046106, 0.066177391789, 0.076638838054, 0.016886573695, 0.002745304761, 0.007667467217, 0.024203691940, 0.036411888726, -0.018288700887, -0.040906084520, -0.059558902032, -0.068063652936, -0.061452924527, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.038720082141, 0.044738571292, 0.046855811675, 0.034625133094, 0.002816976985};
}  // namespace reference

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }

class LayerStudentTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor Features(const Shape& shape, double a, double b) {
        std::vector<float> v(static_cast<size_t>(shape.numel()));
        for (size_t k = 0; k < v.size(); ++k) v[k] = static_cast<float>(std::sin(a * static_cast<double>(k) + b));
        return Tensor(shape, &backend, v);
    }

    Tensor Ids(int64_t n, int64_t length, int64_t vocab) {
        std::vector<float> ids(static_cast<size_t>(n * length));
        for (size_t k = 0; k < ids.size(); ++k) ids[k] = static_cast<float>((7 * k + 3) % static_cast<size_t>(vocab));
        return Tensor(Shape({n, length}), &backend, ids);
    }

    static Tensor* Param(Module& m, const std::string& name) {
        for (const NamedParamRef& p : m.named_parameters()) {
            if (p.name == name) return p.ref.value;
        }
        ADD_FAILURE() << "no parameter " << name;
        return nullptr;
    }
};

TEST_F(LayerStudentTest, EveryOtherLayerTakesOneOutOfTwo) {
    EXPECT_EQ(EveryOtherLayer(5), (std::vector<int64_t>{0, 2, 4}));
    EXPECT_EQ(EveryOtherLayer(30).size(), 15u);
    EXPECT_EQ(EveryOtherLayer(1), (std::vector<int64_t>{0}));
    EXPECT_THROW((void)EveryOtherLayer(0), std::invalid_argument);
}

TEST_F(LayerStudentTest, AStudentWithEveryLayerComputesTheTeachersLogits) {
    for (const std::string name : {"llama", "gemma3", "qwen3"}) {
        std::unique_ptr<CausalLM> teacher = LoadCausalLM(Fixture(name), &backend);
        std::vector<int64_t> all;
        for (int64_t i = 0; i < teacher->num_layers(); ++i) all.push_back(i);
        std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, all, &backend);
        const Tensor ids = Ids(2, 6, teacher->config().vocab_size);
        EXPECT_EQ(student->forward(ids).to_host_vector(), teacher->forward(ids).to_host_vector()) << name;
    }
}

TEST_F(LayerStudentTest, TheStudentCopiesTheChosenLayersAndTheirTypes) {
    std::unique_ptr<CausalLM> teacher = LoadCausalLM(Fixture("gemma3"), &backend);
    ASSERT_EQ(teacher->num_layers(), 4);
    std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, EveryOtherLayer(4), &backend);
    EXPECT_EQ(student->num_layers(), 2);
    EXPECT_EQ(student->config().num_hidden_layers, 2);
    EXPECT_EQ(student->config().layer_types, (std::vector<std::string>{"sliding_attention", "full_attention"}));
    EXPECT_EQ(student->config().is_sliding_layer(0), teacher->config().is_sliding_layer(0));
    EXPECT_EQ(student->config().is_sliding_layer(1), teacher->config().is_sliding_layer(2));
    for (const NamedParamRef& p : student->layer(1).named_parameters()) {
        EXPECT_EQ(p.ref.value->to_host_vector(), Param(teacher->layer(2), p.name)->to_host_vector()) << p.name;
    }
    EXPECT_EQ(Param(*student, "embed_tokens.weight")->to_host_vector(), Param(*teacher, "embed_tokens.weight")->to_host_vector());
    EXPECT_EQ(Param(*student, "norm.weight")->to_host_vector(), Param(*teacher, "norm.weight")->to_host_vector());
}

TEST_F(LayerStudentTest, TheStudentIsASeparateModel) {
    std::unique_ptr<CausalLM> teacher = LoadCausalLM(Fixture("llama"), &backend);
    std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, {1}, &backend);
    const std::vector<float> before = Param(*teacher, "layers.1.mha.q_proj.weight")->to_host_vector();
    Tensor* copy = Param(*student, "layers.0.mha.q_proj.weight");
    EXPECT_EQ(copy->to_host_vector(), before);
    *copy = Tensor(copy->shape(), &backend);
    EXPECT_EQ(Param(*teacher, "layers.1.mha.q_proj.weight")->to_host_vector(), before);
}

TEST_F(LayerStudentTest, BadLayerChoicesAndAdaptedTeachersAreRefused) {
    std::unique_ptr<CausalLM> teacher = LoadCausalLM(Fixture("llama"), &backend);
    EXPECT_THROW((void)MakeStudentFromLayers(*teacher, {}, &backend), std::invalid_argument);
    EXPECT_THROW((void)MakeStudentFromLayers(*teacher, {0, 0}, &backend), std::invalid_argument);
    EXPECT_THROW((void)MakeStudentFromLayers(*teacher, {2}, &backend), std::invalid_argument);
    EXPECT_THROW((void)MakeStudentFromLayers(*teacher, {-1}, &backend), std::invalid_argument);
    (void)ApplyLora(*teacher);
    EXPECT_THROW((void)MakeStudentFromLayers(*teacher, {0}, &backend), std::invalid_argument);
}

TEST_F(LayerStudentTest, TheCosineLossMatchesTheReference) {
    HiddenStateCosineLoss loss(&backend);
    const std::vector<float> mask = {1, 1, 1, 1, 0, 1};
    EXPECT_NEAR(loss.forward(Features(Shape({2, 3, 5}), 0.7, 0.3), Features(Shape({2, 3, 5}), 0.45, 0.1), mask),
                reference::kCosineLoss, 1e-6);
    const std::vector<float> g = loss.backward().to_host_vector();
    ASSERT_EQ(g.size(), reference::kCosineGrad.size());
    for (size_t k = 0; k < g.size(); ++k) EXPECT_NEAR(g[k], reference::kCosineGrad[k], 1e-6) << k;
}

TEST_F(LayerStudentTest, TheCosineLossIsZeroForParallelVectorsAndRefusesBadInput) {
    HiddenStateCosineLoss loss(&backend);
    const Tensor t = Features(Shape({3, 4}), 0.7, 0.3);
    std::vector<float> scaled = t.to_host_vector();
    for (float& v : scaled) v *= 2.5f;
    EXPECT_NEAR(loss.forward(Tensor(Shape({3, 4}), &backend, scaled), t), 0.0, 1e-6);
    for (float v : loss.backward().to_host_vector()) EXPECT_NEAR(v, 0.0, 1e-6);
    // A zero row counts as cos = 0, with no gradient.
    std::vector<float> zero = t.to_host_vector();
    for (size_t k = 0; k < 4; ++k) zero[k] = 0.0f;
    EXPECT_NEAR(loss.forward(Tensor(Shape({3, 4}), &backend, zero), t, {1, 0, 0}), 1.0, 1e-12);
    for (float v : loss.backward().to_host_vector()) EXPECT_EQ(v, 0.0f);
    EXPECT_THROW((void)loss.forward(t, Features(Shape({4, 3}), 0.1, 0.0)), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(Features(Shape({4}), 0.1, 0.0), Features(Shape({4}), 0.1, 0.0)), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(t, t, {1, 1}), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(t, t, {0, 0, 0}), std::invalid_argument);
    HiddenStateCosineLoss fresh(&backend);
    EXPECT_THROW((void)fresh.backward(), std::logic_error);
}

TEST_F(LayerStudentTest, TheTapPassesStatesThroughAndAddsItsGradientOnce) {
    HiddenStateTap tap(&backend);
    EXPECT_THROW((void)tap.hidden(), std::logic_error);
    EXPECT_THROW(tap.add_gradient(Features(Shape({1, 2, 3}), 0.1, 0.0)), std::logic_error);
    const Tensor h = Features(Shape({1, 2, 3}), 0.7, 0.3);
    EXPECT_EQ(tap.forward(h).to_host_vector(), h.to_host_vector());
    EXPECT_EQ(tap.hidden().to_host_vector(), h.to_host_vector());
    const Tensor g = Features(Shape({1, 2, 3}), 0.2, 0.5), extra = Features(Shape({1, 2, 3}), 0.9, 0.1);
    EXPECT_EQ(tap.backward(g).to_host_vector(), g.to_host_vector());
    tap.add_gradient(extra);
    tap.add_gradient(extra);
    const std::vector<float> got = tap.backward(g).to_host_vector();
    const std::vector<float> a = g.to_host_vector(), b = extra.to_host_vector();
    for (size_t k = 0; k < got.size(); ++k) EXPECT_FLOAT_EQ(got[k], a[k] + 2 * b[k]);
    EXPECT_EQ(tap.backward(g).to_host_vector(), a);  // used up
    EXPECT_THROW(tap.add_gradient(Features(Shape({1, 3, 3}), 0.1, 0.0)), std::invalid_argument);
    EXPECT_EQ(tap.named_parameters().size(), 0u);
}

TEST_F(LayerStudentTest, TheTapsGradientIsTheCosineLossesGradientThroughTheModel) {
    std::unique_ptr<CausalLM> teacher = LoadCausalLM(Fixture("llama"), &backend);
    std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, {0}, &backend);
    student->set_intervention(0, std::make_unique<HiddenStateTap>(&backend));
    auto* tap = static_cast<HiddenStateTap*>(student->intervention(0));
    teacher->set_intervention(1, std::make_unique<HiddenStateTap>(&backend));
    auto* teacher_tap = static_cast<HiddenStateTap*>(teacher->intervention(1));
    const Tensor ids = Ids(2, 5, 64);
    (void)teacher->forward(ids);
    const Tensor target = teacher_tap->hidden();
    HiddenStateCosineLoss loss(&backend);
    auto value = [&]() {
        (void)student->forward(ids);
        return static_cast<double>(loss.forward(tap->hidden(), target));
    };
    AdamOptimizer zero(1e-3f, &backend);
    zero.zero_grad(*student);
    const Tensor logits = student->forward(ids);
    (void)loss.forward(tap->hidden(), target);
    tap->add_gradient(loss.backward());
    (void)student->backward(Tensor(logits.shape(), &backend));  // no gradient from the logits
    // The gradient on a few weights of the block against central differences.
    Tensor* w = Param(*student, "layers.0.swiglu.down_proj.weight");
    Tensor* gw = nullptr;
    for (const NamedParamRef& p : student->named_parameters()) {
        if (p.name == "layers.0.swiglu.down_proj.weight") gw = p.ref.grad;
    }
    ASSERT_NE(gw, nullptr);
    const std::vector<float> grad = gw->to_host_vector();
    double largest = 0;
    for (size_t k : {size_t{0}, size_t{17}, size_t{300}, size_t{1000}}) largest = std::max(largest, std::fabs(static_cast<double>(grad[k])));
    EXPECT_GT(largest, 1e-3);  // the check below isn't of zeros
    for (size_t k : {size_t{0}, size_t{17}, size_t{300}, size_t{1000}}) {
        std::vector<float> v = w->to_host_vector();
        const float h = 1e-2f, original = v[k];
        v[k] = original + h;
        *w = Tensor(w->shape(), &backend, v);
        const double up = value();
        v[k] = original - h;
        *w = Tensor(w->shape(), &backend, v);
        const double down = value();
        v[k] = original;
        *w = Tensor(w->shape(), &backend, v);
        EXPECT_NEAR(grad[k], (up - down) / (2 * h), 2e-3 + 0.02 * std::fabs(grad[k])) << k;
    }
}

TEST_F(LayerStudentTest, TrainingOnTheCosineLossAlignsTheStudentsStates) {
    std::unique_ptr<CausalLM> teacher = LoadCausalLM(Fixture("llama"), &backend);
    std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, {0}, &backend);
    student->set_intervention(0, std::make_unique<HiddenStateTap>(&backend));
    auto* tap = static_cast<HiddenStateTap*>(student->intervention(0));
    teacher->set_intervention(1, std::make_unique<HiddenStateTap>(&backend));
    const Tensor ids = Ids(4, 6, 64);
    (void)teacher->forward(ids);
    const Tensor target = static_cast<HiddenStateTap*>(teacher->intervention(1))->hidden();
    HiddenStateCosineLoss loss(&backend);
    AdamOptimizer opt(1e-2f, &backend);
    float first = 0, last = 0;
    for (int step = 0; step < 40; ++step) {
        opt.zero_grad(*student);
        const Tensor logits = student->forward(ids);
        last = loss.forward(tap->hidden(), target);
        if (step == 0) first = last;
        tap->add_gradient(loss.backward());
        (void)student->backward(Tensor(logits.shape(), &backend));
        opt.step(*student);
    }
    EXPECT_LT(last, 0.5f * first) << first << " -> " << last;
}

}  // namespace
}  // namespace pulsatrix
