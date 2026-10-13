// XAI-1: the nine question categories, the four reasons, and every explainer's declaration of what
// it answers.
#include "pulsatrix/explainer_capabilities.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/pdp.hpp"
#include "pulsatrix/protein_explanations.hpp"
#include "pulsatrix/saliency.hpp"

namespace pulsatrix {
namespace {

std::string ReadHeader(const std::string& header) {
    std::ifstream in(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/../../include/" + header);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

// Whether @p text contains @p word with no identifier character on either side.
bool ContainsWord(const std::string& text, const std::string& word) {
    const auto ident = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    for (std::size_t at = text.find(word); at != std::string::npos; at = text.find(word, at + 1)) {
        const std::size_t end = at + word.size();
        if ((at == 0 || !ident(text[at - 1])) && (end == text.size() || !ident(text[end]))) return true;
    }
    return false;
}

std::vector<std::string> Names(const std::vector<const ExplainerCapabilities*>& explainers) {
    std::vector<std::string> names;
    for (const ExplainerCapabilities* e : explainers) names.push_back(e->name);
    return names;
}

TEST(XaiQuestions, NineQuestionsAndFourReasonsRoundTripThroughTheirNames) {
    ASSERT_EQ(AllXaiQuestions().size(), 9u);
    ASSERT_EQ(AllXaiReasons().size(), 4u);
    std::set<std::string_view> names;
    for (XaiQuestion q : AllXaiQuestions()) {
        EXPECT_EQ(ParseXaiQuestion(to_string(q)), q);
        EXPECT_TRUE(names.insert(to_string(q)).second);
        const std::string_view text = QuestionText(q);
        ASSERT_FALSE(text.empty());
        EXPECT_EQ(text.back(), '?');
    }
    for (XaiReason r : AllXaiReasons()) {
        EXPECT_EQ(ParseXaiReason(to_string(r)), r);
        EXPECT_FALSE(ReasonText(r).empty());
    }
    EXPECT_EQ(to_string(XaiQuestion::HowToStillBeThis), "how_to_still_be_this");
    EXPECT_EQ(to_string(XaiReason::Discover), "discover");
    EXPECT_EQ(ParseXaiQuestion("Why"), std::nullopt);  // names are case-sensitive
    EXPECT_EQ(ParseXaiQuestion("why not"), std::nullopt);
    EXPECT_EQ(ParseXaiReason(""), std::nullopt);
}

TEST(ExplainerCatalog, EveryEntryIsCompleteAndNamedOnce) {
    std::set<std::string> names;
    for (const ExplainerCapabilities& e : ExplainerCatalog()) {
        SCOPED_TRACE(e.name);
        EXPECT_TRUE(names.insert(e.name).second);
        EXPECT_FALSE(e.summary.empty());
        EXPECT_FALSE(e.entry_points.empty());
        ASSERT_FALSE(e.questions.empty());
        EXPECT_EQ(std::set<XaiQuestion>(e.questions.begin(), e.questions.end()).size(), e.questions.size());
        EXPECT_EQ(&ExplainerCapabilitiesOf(e.name), &e);
    }
    EXPECT_THROW((void)ExplainerCapabilitiesOf("SmoothGrad"), std::invalid_argument);
}

// The declarations name real code: each header exists and declares each entry point.
TEST(ExplainerCatalog, EntryPointsAreDeclaredInTheirHeaders) {
    for (const ExplainerCapabilities& e : ExplainerCatalog()) {
        SCOPED_TRACE(e.name);
        const std::string text = ReadHeader(e.header);
        ASSERT_FALSE(text.empty()) << e.header;
        for (const std::string& entry : e.entry_points) {
            const std::string symbol = entry.substr(entry.rfind(':') == std::string::npos ? 0 : entry.rfind(':') + 1);
            EXPECT_TRUE(ContainsWord(text, symbol)) << entry << " in " << e.header;
        }
    }
}

TEST(ExplainerCatalog, ExplainerClassesReturnTheirOwnEntry) {
    EXPECT_EQ(&Saliency::capabilities(), &ExplainerCapabilitiesOf("Saliency"));
    EXPECT_EQ(IntegratedGradients::capabilities().name, "IntegratedGradients");
    EXPECT_EQ(GradCAM::capabilities().input, InputKind::Image);
    EXPECT_EQ(KernelSHAP::capabilities().access, ModelAccess::BlackBox);
    EXPECT_EQ(LIME::capabilities().scope, ExplanationScope::Local);
    EXPECT_EQ(PDP::capabilities().scope, ExplanationScope::Global);
    EXPECT_TRUE(LRP::capabilities().answers(XaiQuestion::WhyNot));
    EXPECT_EQ(EncoderExplainer::capabilities().input, InputKind::Protein);
}

TEST(ExplainerCatalog, ExplainersForPutsThePrimaryAnswersFirst) {
    const std::vector<std::string> why = Names(ExplainersFor(XaiQuestion::Why));
    ASSERT_GE(why.size(), 2u);
    EXPECT_EQ(why.front(), "LRP");
    // Occlusion answers What if second, so it follows every explainer whose first answer is What if.
    const std::vector<std::string> what_if = Names(ExplainersFor(XaiQuestion::WhatIf));
    const auto at = [&](const std::string& n) { return std::find(what_if.begin(), what_if.end(), n) - what_if.begin(); };
    EXPECT_LT(at("LocalSensitivity"), at("Occlusion"));
    EXPECT_LT(at("ICE"), at("PDP"));
    EXPECT_EQ(Names(ExplainersFor(XaiQuestion::HowToBeThat)),
              (std::vector<std::string>{"GradientCounterfactual", "GrowingSpheres", "DiverseCounterfactuals"}));
    // LRP's contrastive target answers Why not before the counterfactuals do.
    EXPECT_EQ(Names(ExplainersFor(XaiQuestion::WhyNot)).front(), "LRP");
    EXPECT_EQ(Names(ExplainersFor(XaiQuestion::Input)), (std::vector<std::string>{"NearestNeighbors"}));
}

// The gaps the roadmap lists: no explainer yet for these. XAI-4 (Anchors) and KS-3 (uncertainty)
// fill the first two; when they land, this test changes with them.
TEST(ExplainerCatalog, QuestionsWithNoExplainerYetComeBackEmpty) {
    EXPECT_TRUE(ExplainersFor(XaiQuestion::HowToStillBeThis).empty());
    EXPECT_TRUE(ExplainersFor(XaiQuestion::Performance).empty());
    EXPECT_TRUE(ExplainersFor(XaiQuestion::Output).empty());
}

TEST(ExplainerCatalog, TheDocumentRoundTripsThroughJson) {
    const JsonValue doc = ParseJson(WriteJson(ExplainerCatalogDocument()));
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.explainer_catalog.v1");
    const JsonValue::Array& explainers = doc.find("explainers")->as_array();
    ASSERT_EQ(explainers.size(), ExplainerCatalog().size());
    for (std::size_t i = 0; i < explainers.size(); ++i) {
        const ExplainerCapabilities& e = ExplainerCatalog()[i];
        const JsonValue& j = explainers[i];
        EXPECT_EQ(j.find("name")->as_string(), e.name);
        EXPECT_EQ(j.find("header")->as_string(), e.header);
        EXPECT_EQ(j.find("access")->as_string(), to_string(e.access));
        const JsonValue::Array& questions = j.find("questions")->as_array();
        ASSERT_EQ(questions.size(), e.questions.size());
        for (std::size_t q = 0; q < questions.size(); ++q) {
            EXPECT_EQ(ParseXaiQuestion(questions[q].as_string()), e.questions[q]);
        }
    }
    const JsonValue& lrp = explainers.front();
    EXPECT_EQ(lrp.find("scope")->as_string(), "local");
    EXPECT_EQ(lrp.find("input")->as_string(), "any");
    EXPECT_EQ(lrp.find("entry_points")->as_array().front().as_string(), "LRP::explain");
    EXPECT_TRUE(lrp.find("complete")->as_bool());
    EXPECT_FALSE(lrp.find("needs_base_model")->as_bool());
}

TEST(ExplainerCatalog, CompletenessAndBaseModelsAreDeclared) {
    EXPECT_TRUE(IntegratedGradients::capabilities().complete);
    EXPECT_TRUE(KernelSHAP::capabilities().complete);
    EXPECT_FALSE(Saliency::capabilities().complete);  // a gradient doesn't add up to the output
    EXPECT_FALSE(LIME::capabilities().complete);
    EXPECT_TRUE(ExplainerCapabilitiesOf("FineTuneRelevanceDiff").needs_base_model);
    EXPECT_TRUE(ExplainerCapabilitiesOf("IntruderDimensions").needs_base_model);
    for (const ExplainerCapabilities& e : ExplainerCatalog()) {
        // Completeness is a property of attributions: of local explainers that answer Why.
        if (e.complete) EXPECT_TRUE(e.answers(XaiQuestion::Why)) << e.name;
        // An explainer declared to return an Attribution does so in its header.
        if (e.attribution) EXPECT_TRUE(ContainsWord(ReadHeader(e.header), "Attribution")) << e.name;
    }
    EXPECT_TRUE(ExplainerCapabilitiesOf("Occlusion").attribution);
    EXPECT_FALSE(ExplainerCapabilitiesOf("LogitLens").attribution);
}

}  // namespace
}  // namespace pulsatrix
