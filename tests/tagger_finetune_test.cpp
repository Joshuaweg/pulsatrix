#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tagger_finetune_example.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

class TaggerFineTuneTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::string path = ::testing::TempDir() + "pulsatrix_tagger_" +
                       ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".safetensors";
    void TearDown() override {
        std::remove(path.c_str());
        std::remove(OptimizerStatePath(path).c_str());
    }
};

TEST_F(TaggerFineTuneTest, TaggerMapsTokenSequencesToPerTokenLogits) {
    TinyTagger tagger(&backend);
    InitTagger(tagger, 1);
    TaggingBatch batch = MakeTaggingBatch(TaggingRule::SumWithPrevious, 4, 7);
    Tensor logits = tagger.forward(batch.inputs);
    EXPECT_EQ(logits.shape(), Shape({4, TinyTagger::kSeqLen, TinyTagger::kClasses}));
    // Names compose through the embedding, the block and the head.
    std::vector<NamedParamRef> params = tagger.named_parameters();
    EXPECT_EQ(params.front().name, "embed.weight");
    EXPECT_EQ(params.back().name, "head.bias");
}

// The tagger's own plumbing (reshapes between the layers) checked against finite differences on
// a few parameters, through the real loss.
TEST_F(TaggerFineTuneTest, GradientsMatchFiniteDifferences) {
    TinyTagger tagger(&backend);
    InitTagger(tagger, 2);
    TaggingBatch batch = MakeTaggingBatch(TaggingRule::SumWithPrevious, 3, 11);
    auto loss_value = [&] {
        TokenCrossEntropyLoss loss(&backend);
        return static_cast<double>(loss.forward(tagger.forward(batch.inputs), batch.targets));
    };
    for (ParamRef p : tagger.parameters()) p.grad->fill(0.0f);
    TokenCrossEntropyLoss loss(&backend);
    (void)loss.forward(tagger.forward(batch.inputs), batch.targets);
    (void)tagger.backward(loss.backward());

    std::vector<NamedParamRef> params = tagger.named_parameters();
    for (const NamedParamRef& p : {params.front(), params[params.size() / 2], params.back()}) {
        for (int64_t i : {int64_t{0}, p.ref.value->numel() / 2}) {
            float* v = p.ref.value->data() + i;
            const float saved = *v;
            const float h = 1e-2f;
            *v = saved + h;
            const double up = loss_value();
            *v = saved - h;
            const double down = loss_value();
            *v = saved;
            const double numeric = (up - down) / (2.0 * h);
            EXPECT_NEAR(p.ref.grad->data()[i], numeric, 2e-3 + 2e-2 * std::fabs(numeric)) << p.name << "[" << i << "]";
        }
    }
}

// The recipe end to end: pretrain on rule A, save, reload, fully fine-tune on rule B.
TEST_F(TaggerFineTuneTest, PretrainSaveReloadAndFineTune) {
    FineTuneConfig pretrain_config;
    pretrain_config.steps = 200;
    TinyTagger pretrained(&backend);
    InitTagger(pretrained, 3);
    const std::vector<float> curve_a = TrainTagger(pretrained, TaggingRule::SumWithPrevious, pretrain_config, &backend);
    EXPECT_LT(curve_a.back(), 0.5f * curve_a.front()) << "pretraining should learn rule A";
    SaveCheckpoint(path, pretrained);

    TinyTagger tuned(&backend);
    LoadCheckpoint(path, tuned);
    const float before = TaggingAccuracy(tuned, TaggingRule::DifferenceWithPrevious, 500);
    FineTuneConfig finetune_config;
    finetune_config.steps = 200;
    const std::vector<float> curve_b = TrainTagger(tuned, TaggingRule::DifferenceWithPrevious, finetune_config, &backend);
    const float after = TaggingAccuracy(tuned, TaggingRule::DifferenceWithPrevious, 500);
    EXPECT_LT(curve_b.back(), curve_b.front());
    EXPECT_GT(after, before + 0.2f) << "fine-tuning should learn rule B";
}

// Interrupting fine-tuning and resuming from a checkpoint (model, AdamW state, scheduler step)
// reproduces the uninterrupted run exactly.
TEST_F(TaggerFineTuneTest, ResumingMidFineTuneReproducesTheRun) {
    FineTuneConfig config;
    config.steps = 12;
    TinyTagger straight(&backend);
    InitTagger(straight, 4);
    const std::vector<float> full = TrainTagger(straight, TaggingRule::DifferenceWithPrevious, config, &backend);

    TinyTagger first(&backend);
    InitTagger(first, 4);
    FineTuneConfig half = config;
    half.stop_after = 5;
    half.checkpoint_path = path;
    const std::vector<float> part1 = TrainTagger(first, TaggingRule::DifferenceWithPrevious, half, &backend);

    TinyTagger second(&backend);
    FineTuneConfig rest = config;
    rest.resume_from = path;
    const std::vector<float> part2 = TrainTagger(second, TaggingRule::DifferenceWithPrevious, rest, &backend);

    std::vector<float> joined = part1;
    joined.insert(joined.end(), part2.begin(), part2.end());
    EXPECT_EQ(joined, full);
    EXPECT_EQ(values_of(*second.named_parameters().back().ref.value),
              values_of(*straight.named_parameters().back().ref.value));
}

TEST_F(TaggerFineTuneTest, BatchesAreRaggedAndPadded) {
    TaggingBatch batch = MakeTaggingBatch(TaggingRule::SumWithPrevious, 16, 3);
    const int64_t tokens = CountTargetTokens(batch.targets);
    EXPECT_LT(tokens, 16 * TinyTagger::kSeqLen);  // some positions are padding
    EXPECT_GT(tokens, 16 * TinyTagger::kSeqLen / 2);
}

}  // namespace
}  // namespace pulsatrix
