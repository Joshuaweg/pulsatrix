// pulsatrix_plan: which explainers answer a question about a model, and which checks they need
// (roadmap XAI-2). It runs nothing; it prints ExplanationPlanner's plan.
//
//   pulsatrix_plan --question why|why_not|what_if|how_global|how_to_be_that|how_to_still_be_this|
//                             input|output|performance
//                  --reason justify|control|improve|discover
//                  [--access black_box|weights|activations|gradients|relevance]
//                  [--input any|tabular|image|tokens|protein|symbolic|representation]
//                  [--reinitialize] [--dataset] [--labels] [--base-model] [--features N] [--json]
//
// The defaults are the least a caller can have: a black box, input of unknown kind, one input.
// --reinitialize: the model is a pulsatrix Module whose weights can be redrawn and restored.
// --dataset: there are many inputs, not just the one being explained.
// --labels: the dataset has true labels and a slice for each input, for per-slice error; implies --dataset.
// --base-model: the model it was fine-tuned from is at hand too.
// --features N: each input has N features; explainers with a lower limit (KernelSHAP: 20) drop out.
// --json prints a pulsatrix.explanation_plan.v1 document instead of text.
// Exit status: 0 for a complete plan, 1 when a warning says something can't be delivered, 2 for
// a usage error.
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

#include "pulsatrix/explanation_planner.hpp"

using namespace pulsatrix;

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_plan: " << problem << "\n"
              << "usage: pulsatrix_plan --question QUESTION --reason REASON [--access ACCESS] [--input INPUT]\n"
              << "                      [--reinitialize] [--dataset] [--labels] [--base-model] [--features N] [--json]\n"
              << "  QUESTION: input output performance how_global why why_not what_if how_to_be_that how_to_still_be_this\n"
              << "  REASON:   justify control improve discover\n"
              << "  ACCESS:   black_box weights activations gradients relevance (default black_box)\n"
              << "  INPUT:    any tabular image tokens protein symbolic representation (default any)\n";
    std::exit(2);
}

template <typename Enum>
std::optional<Enum> ParseName(const std::string& text, std::initializer_list<Enum> values) {
    for (Enum v : values) {
        if (to_string(v) == text) return v;
    }
    return std::nullopt;
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<XaiQuestion> question;
    std::optional<XaiReason> reason;
    ModelTraits traits;
    bool json = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--question") {
            const std::string v = value();
            question = ParseXaiQuestion(v);
            if (!question) Usage("unknown question \"" + v + "\"");
        } else if (a == "--reason") {
            const std::string v = value();
            reason = ParseXaiReason(v);
            if (!reason) Usage("unknown reason \"" + v + "\"");
        } else if (a == "--access") {
            const std::string v = value();
            const auto access = ParseName(v, {ModelAccess::BlackBox, ModelAccess::Weights, ModelAccess::Activations,
                                              ModelAccess::Gradients, ModelAccess::Relevance});
            if (!access) Usage("unknown access \"" + v + "\"");
            traits.access = *access;
        } else if (a == "--input") {
            const std::string v = value();
            const auto input = ParseName(v, {InputKind::Any, InputKind::Tabular, InputKind::Image, InputKind::Tokens,
                                             InputKind::Protein, InputKind::Symbolic, InputKind::Representation});
            if (!input) Usage("unknown input \"" + v + "\"");
            traits.input = *input;
        } else if (a == "--reinitialize") {
            traits.can_reinitialize = true;
        } else if (a == "--dataset") {
            traits.has_dataset = true;
        } else if (a == "--labels") {
            traits.has_dataset = true;
            traits.has_labels = true;
        } else if (a == "--base-model") {
            traits.has_base_model = true;
        } else if (a == "--features") {
            const std::string v = value();
            try {
                traits.features = std::stoll(v);
            } catch (const std::exception&) {
                Usage("--features needs a number, not \"" + v + "\"");
            }
            if (traits.features < 0) Usage("--features must not be negative");
        } else if (a == "--json") {
            json = true;
        } else {
            Usage("unknown option " + a);
        }
    }
    if (!question) Usage("needs --question");
    if (!reason) Usage("needs --reason");
    const ExplanationPlan plan = ExplanationPlanner().plan(*question, *reason, traits);
    std::cout << (json ? WriteJson(ToJson(plan)) + "\n" : DescribePlan(plan));
    return plan.complete() ? 0 : 1;
}
