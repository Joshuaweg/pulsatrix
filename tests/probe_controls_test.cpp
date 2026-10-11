// INT-3: Hewitt and Liang's control task and selectivity, and a probe compared against a baseline
// representation.
#include "pulsatrix/probe_controls.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

/** @brief A seeded LCG in [0, 1), the same on every platform. */
struct Lcg {
    uint64_t s;
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967296.0f;
    }
    float gaussian() {
        // Sum of 12 uniforms: mean 6, variance 1. Good enough for test data.
        float sum = 0;
        for (int k = 0; k < 12; ++k) sum += next();
        return sum - 6.0f;
    }
};

TEST(ControlTaskLabels, EachTypeKeepsOneLabelWhateverTheRowOrder) {
    std::vector<int64_t> types;
    std::vector<int> labels;
    for (int64_t i = 0; i < 300; ++i) {
        types.push_back((i * 37) % 50 + 1000);  // 50 types, six rows each, interleaved
        labels.push_back(static_cast<int>(i % 3));
    }
    const std::vector<int> control = ControlTaskLabels(types, labels, 3, 11);
    std::map<int64_t, int> by_type;
    for (size_t i = 0; i < types.size(); ++i) {
        const auto [it, inserted] = by_type.emplace(types[i], control[i]);
        if (!inserted) EXPECT_EQ(it->second, control[i]) << "type " << types[i];
    }
    EXPECT_EQ(by_type.size(), 50u);

    // The same rows in reverse order give each type the same label.
    const std::vector<int64_t> reversed_types(types.rbegin(), types.rend());
    const std::vector<int> reversed_labels(labels.rbegin(), labels.rend());
    const std::vector<int> reversed = ControlTaskLabels(reversed_types, reversed_labels, 3, 11);
    for (size_t i = 0; i < reversed.size(); ++i) EXPECT_EQ(reversed[i], by_type.at(reversed_types[i]));

    EXPECT_EQ(ControlTaskLabels(types, labels, 3, 11), control);
    EXPECT_NE(ControlTaskLabels(types, labels, 3, 12), control);
}

TEST(ControlTaskLabels, FollowsTheLabelsDistribution) {
    // 4,000 types, one row each, a quarter of them labeled 1 and none labeled 2.
    std::vector<int64_t> types;
    std::vector<int> labels;
    for (int64_t i = 0; i < 4000; ++i) {
        types.push_back(i);
        labels.push_back(i % 4 == 0 ? 1 : 0);
    }
    const std::vector<int> control = ControlTaskLabels(types, labels, 3, 5);
    const auto ones = std::count(control.begin(), control.end(), 1);
    EXPECT_NEAR(static_cast<double>(ones) / 4000.0, 0.25, 0.02);
    EXPECT_EQ(std::count(control.begin(), control.end(), 2), 0);
    // The control isn't the task: about 0.75^2 + 0.25^2 = 0.625 of the rows agree by chance.
    int64_t agree = 0;
    for (size_t i = 0; i < labels.size(); ++i) agree += control[i] == labels[i] ? 1 : 0;
    EXPECT_NEAR(static_cast<double>(agree) / 4000.0, 0.625, 0.03);
}

TEST(ControlTaskLabels, RejectsMalformedInput) {
    EXPECT_THROW((void)ControlTaskLabels({1, 2}, {0}, 2, 0), std::invalid_argument);
    EXPECT_THROW((void)ControlTaskLabels({}, {}, 2, 0), std::invalid_argument);
    EXPECT_THROW((void)ControlTaskLabels({1}, {0}, 1, 0), std::invalid_argument);
    EXPECT_THROW((void)ControlTaskLabels({1, 2}, {0, 2}, 2, 0), std::invalid_argument);
    EXPECT_THROW((void)ControlTaskLabels({1, 2}, {0, -1}, 2, 0), std::invalid_argument);
}

/** @brief Rows of 100 token types, 20 rows each, where each type's label is a linear function of a
 *         hidden 8-dimensional vector for that type, as a part of speech is a property of a word. */
struct TypedData {
    static constexpr int64_t kTypes = 100, kPerType = 20, kHidden = 8;
    int64_t rows = kTypes * kPerType;
    std::vector<int64_t> types;
    std::vector<int> labels;
    std::vector<bool> train, test;
    std::vector<float> structured;  ///< the hidden vector plus noise: rows x 8
    std::vector<float> one_hot;     ///< the type's identity plus noise: rows x 100

    TypedData() {
        Lcg rng{3};
        std::vector<float> hidden(static_cast<size_t>(kTypes * kHidden));
        for (float& v : hidden) v = rng.gaussian();
        std::vector<int> type_label(static_cast<size_t>(kTypes));
        for (int64_t t = 0; t < kTypes; ++t) {
            float projection = 0;
            for (int64_t j = 0; j < kHidden; ++j) projection += hidden[static_cast<size_t>(t * kHidden + j)] * (j % 2 == 0 ? 1.0f : -0.5f);
            type_label[static_cast<size_t>(t)] = projection > 0 ? 1 : 0;
        }
        for (int64_t r = 0; r < rows; ++r) {
            // Every type has 16 training rows and 4 test rows, so the test holds no unseen type.
            const int64_t t = r % kTypes;
            types.push_back(t);
            labels.push_back(type_label[static_cast<size_t>(t)]);
            train.push_back((r / kTypes) % 5 != 0);
            test.push_back((r / kTypes) % 5 == 0);
            for (int64_t j = 0; j < kHidden; ++j) structured.push_back(hidden[static_cast<size_t>(t * kHidden + j)] + 0.2f * rng.gaussian());
            for (int64_t j = 0; j < kTypes; ++j) one_hot.push_back((j == t ? 1.0f : 0.0f) + 0.05f * rng.gaussian());
        }
    }
};

ProbeOptions Options() {
    ProbeOptions o;
    o.epochs = 60;
    o.batch = 128;
    o.seed = 1;
    return o;
}

// The falsifier, Hewitt and Liang's finding in miniature. Both representations let a linear probe
// read the task, but only the structured one is selective: the one-hot representation lets the
// probe memorize a label for every type, so it learns the control task as well as the real one.
TEST(MeasureSelectivity, SeparatesAStructuredRepresentationFromOneThatOnlyNamesTheType) {
    CPUBackend cpu;
    const TypedData d;
    const ProbeSelectivity structured =
        MeasureSelectivity(d.structured, TypedData::kHidden, d.labels, 2, d.types, d.train, d.test, &cpu, Options());
    const ProbeSelectivity one_hot =
        MeasureSelectivity(d.one_hot, TypedData::kTypes, d.labels, 2, d.types, d.train, d.test, &cpu, Options());

    EXPECT_GT(structured.task.accuracy, 0.9);
    EXPECT_GT(one_hot.task.accuracy, 0.9);
    EXPECT_LT(structured.control.accuracy, 0.8);
    EXPECT_GT(one_hot.control.accuracy, 0.9);
    EXPECT_GT(structured.selectivity, 0.2);
    EXPECT_LT(one_hot.selectivity, 0.1);
    EXPECT_DOUBLE_EQ(structured.selectivity, structured.task.accuracy - structured.control.accuracy);
    EXPECT_DOUBLE_EQ(structured.balanced_selectivity, structured.task.balanced_accuracy - structured.control.balanced_accuracy);
    EXPECT_EQ(structured.train_types, TypedData::kTypes);
    EXPECT_EQ(structured.task.train, 1600);
    EXPECT_EQ(structured.task.test, 400);
}

TEST(MeasureSelectivity, TheTaskProbeIsTheOrdinaryProbe) {
    CPUBackend cpu;
    const TypedData d;
    ProbeOptions o = Options();
    o.epochs = 5;
    const ProbeSelectivity s = MeasureSelectivity(d.structured, TypedData::kHidden, d.labels, 2, d.types, d.train, d.test, &cpu, o);
    const ProbeResult direct = TrainLinearProbe(d.structured, TypedData::kHidden, d.labels, 2, d.train, d.test, &cpu, o);
    EXPECT_EQ(s.task.accuracy, direct.accuracy);
    EXPECT_EQ(s.task.auc, direct.auc);
    const std::vector<int> control = ControlTaskLabels(d.types, d.labels, 2, o.seed);
    const ProbeResult direct_control = TrainLinearProbe(d.structured, TypedData::kHidden, control, 2, d.train, d.test, &cpu, o);
    EXPECT_EQ(s.control.accuracy, direct_control.accuracy);

    const std::vector<int64_t> too_few(d.types.begin(), d.types.end() - 1);
    EXPECT_THROW((void)MeasureSelectivity(d.structured, TypedData::kHidden, d.labels, 2, too_few, d.train, d.test, &cpu, o),
                 std::invalid_argument);
}

/** @brief rows x 4 uniform features; the baseline is the first two columns. */
struct BaselineData {
    int64_t rows = 2000;
    std::vector<float> features, baseline;
    std::vector<int> from_extra, from_baseline;
    std::vector<bool> train, test;

    BaselineData() {
        Lcg rng{9};
        for (int64_t r = 0; r < rows; ++r) {
            float x[4];
            for (float& v : x) v = rng.next() * 2 - 1;
            features.insert(features.end(), x, x + 4);
            baseline.insert(baseline.end(), x, x + 2);
            from_extra.push_back(x[2] - 0.5f * x[3] > 0 ? 1 : 0);
            from_baseline.push_back(x[0] + 0.5f * x[1] > 0 ? 1 : 0);
            train.push_back(r % 5 != 0);
            test.push_back(r % 5 == 0);
        }
    }
};

TEST(CompareToBaseline, GainsOnlyWhatTheBaselineLacks) {
    CPUBackend cpu;
    const BaselineData d;
    ProbeOptions o = Options();
    o.epochs = 40;
    // The label lives in the columns the baseline doesn't have.
    const ProbeGain extra = CompareToBaseline(d.features, 4, d.baseline, 2, d.from_extra, 2, d.train, d.test, &cpu, o);
    EXPECT_GT(extra.probe.accuracy, 0.95);
    EXPECT_LT(extra.baseline.accuracy, 0.65);
    EXPECT_GT(extra.gain, 0.3);
    EXPECT_GT(extra.auc_gain, 0.3);
    EXPECT_DOUBLE_EQ(extra.gain, extra.probe.accuracy - extra.baseline.accuracy);
    EXPECT_DOUBLE_EQ(extra.balanced_gain, extra.probe.balanced_accuracy - extra.baseline.balanced_accuracy);

    // The label lives in the baseline's columns: the representation adds nothing.
    const ProbeGain none = CompareToBaseline(d.features, 4, d.baseline, 2, d.from_baseline, 2, d.train, d.test, &cpu, o);
    EXPECT_GT(none.baseline.accuracy, 0.95);
    EXPECT_LT(std::abs(none.gain), 0.03);

    // Against itself the gain is exactly zero: the two probes train identically.
    const ProbeGain self = CompareToBaseline(d.features, 4, d.features, 4, d.from_extra, 2, d.train, d.test, &cpu, o);
    EXPECT_EQ(self.gain, 0.0);
    EXPECT_EQ(self.auc_gain, 0.0);
}

TEST(CompareToBaseline, HasNoAucGainForMoreThanTwoClassesAndRejectsMismatchedRows) {
    CPUBackend cpu;
    const BaselineData d;
    ProbeOptions o = Options();
    o.epochs = 2;
    std::vector<int> three(d.from_extra);
    for (size_t i = 0; i < three.size(); ++i) three[i] += d.from_baseline[i];
    const ProbeGain g = CompareToBaseline(d.features, 4, d.baseline, 2, three, 3, d.train, d.test, &cpu, o);
    EXPECT_TRUE(std::isnan(g.auc_gain));

    const std::vector<float> short_baseline(d.baseline.begin(), d.baseline.end() - 2);
    EXPECT_THROW((void)CompareToBaseline(d.features, 4, short_baseline, 2, d.from_extra, 2, d.train, d.test, &cpu, o),
                 std::invalid_argument);
    EXPECT_THROW((void)CompareToBaseline(d.features, 4, d.baseline, 0, d.from_extra, 2, d.train, d.test, &cpu, o), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
