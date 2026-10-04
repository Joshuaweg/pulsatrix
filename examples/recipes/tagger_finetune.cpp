/** @file tagger_finetune.cpp
 *  @brief Recipe: full fine-tuning (roadmap TRN-6). Pretrains a tiny transformer tagger on one
 *         rule, saves it, reloads it and fine-tunes every parameter on a related rule, with the
 *         whole v1.1 training stack. Paired with docs/recipes/deep-learning/tagger_finetune.md.
 *
 *   ./tagger_finetune_recipe [steps] [learning_rate]
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tagger_finetune_example.hpp"

using namespace pulsatrix;

namespace {

void print_curve(const char* label, const std::vector<float>& losses) {
    std::printf("%s loss:", label);
    for (size_t i = 0; i < losses.size(); i += losses.size() / 5 == 0 ? 1 : losses.size() / 5) {
        std::printf("  step %zu: %.3f", i, losses[i]);
    }
    std::printf("  final: %.3f\n", losses.back());
}

}  // namespace

int main(int argc, char** argv) {
    CPUBackend backend;
    FineTuneConfig config;
    if (argc > 1) config.steps = std::atoll(argv[1]);
    if (argc > 2) config.learning_rate = static_cast<float>(std::atof(argv[2]));
    const std::string path = "tagger_pretrained.safetensors";

    // 1. "Pretrain" on rule A: tag = (token + previous token) mod 4.
    TinyTagger pretrained(&backend);
    InitTagger(pretrained, 3);
    print_curve("pretrain (rule A)", TrainTagger(pretrained, TaggingRule::SumWithPrevious, config, &backend));
    std::printf("rule A accuracy after pretraining: %.3f\n",
                TaggingAccuracy(pretrained, TaggingRule::SumWithPrevious, 500));
    SaveCheckpoint(path, pretrained);

    // 2. Reload and fully fine-tune on rule B: tag = (token - previous token) mod 4.
    TinyTagger tuned(&backend);
    LoadCheckpoint(path, tuned);
    std::printf("rule B accuracy before fine-tuning: %.3f\n", TaggingAccuracy(tuned, TaggingRule::DifferenceWithPrevious, 500));
    print_curve("fine-tune (rule B)", TrainTagger(tuned, TaggingRule::DifferenceWithPrevious, config, &backend));
    std::printf("rule B accuracy after fine-tuning:  %.3f\n", TaggingAccuracy(tuned, TaggingRule::DifferenceWithPrevious, 500));

    // 3. The baseline every other method is compared against: the same budget from scratch.
    TinyTagger scratch(&backend);
    InitTagger(scratch, 3);
    (void)TrainTagger(scratch, TaggingRule::DifferenceWithPrevious, config, &backend);
    std::printf("rule B accuracy from scratch, same budget: %.3f\n",
                TaggingAccuracy(scratch, TaggingRule::DifferenceWithPrevious, 500));
    return 0;
}
