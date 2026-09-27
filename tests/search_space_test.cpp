#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "pulsatrix/search_space.hpp"

namespace pulsatrix {
namespace {

TEST(SearchSpaceTest, AddContinuousStoresBounds) {
    SearchSpace space;
    space.AddContinuous("learning_rate", 0.001, 0.1);
    ASSERT_EQ(space.size(), 1u);
    const auto& spec = space.Get("learning_rate");
    EXPECT_EQ(spec.name, "learning_rate");
    EXPECT_EQ(spec.kind, ParameterKind::Continuous);
    EXPECT_DOUBLE_EQ(spec.lower, 0.001);
    EXPECT_DOUBLE_EQ(spec.upper, 0.1);
}

TEST(SearchSpaceTest, AddLogUniformStoresBounds) {
    SearchSpace space;
    space.AddLogUniform("learning_rate", 1e-4, 1.0);
    const auto& spec = space.Get("learning_rate");
    EXPECT_EQ(spec.kind, ParameterKind::LogUniform);
    EXPECT_DOUBLE_EQ(spec.lower, 1e-4);
    EXPECT_DOUBLE_EQ(spec.upper, 1.0);
}

TEST(SearchSpaceTest, AddIntegerStoresBounds) {
    SearchSpace space;
    space.AddInteger("batch_size", 8, 128);
    const auto& spec = space.Get("batch_size");
    EXPECT_EQ(spec.kind, ParameterKind::Integer);
    EXPECT_DOUBLE_EQ(spec.lower, 8.0);
    EXPECT_DOUBLE_EQ(spec.upper, 128.0);
}

TEST(SearchSpaceTest, AddCategoricalStoresCategories) {
    SearchSpace space;
    space.AddCategorical("activation", {"relu", "tanh", "sigmoid"});
    const auto& spec = space.Get("activation");
    EXPECT_EQ(spec.kind, ParameterKind::Categorical);
    EXPECT_EQ(spec.categories, (std::vector<std::string>{"relu", "tanh", "sigmoid"}));
}

TEST(SearchSpaceTest, ParametersPreservesInsertionOrder) {
    SearchSpace space;
    space.AddContinuous("a", 0.0, 1.0);
    space.AddInteger("b", 1, 2);
    space.AddCategorical("c", {"x"});
    ASSERT_EQ(space.parameters().size(), 3u);
    EXPECT_EQ(space.parameters()[0].name, "a");
    EXPECT_EQ(space.parameters()[1].name, "b");
    EXPECT_EQ(space.parameters()[2].name, "c");
}

TEST(SearchSpaceTest, ContainsReflectsAddedParameters) {
    SearchSpace space;
    space.AddContinuous("a", 0.0, 1.0);
    EXPECT_TRUE(space.Contains("a"));
    EXPECT_FALSE(space.Contains("nonexistent"));
}

TEST(SearchSpaceTest, GetThrowsOnUnknownName) {
    SearchSpace space;
    space.AddContinuous("a", 0.0, 1.0);
    EXPECT_THROW((void)space.Get("b"), std::out_of_range);
}

TEST(SearchSpaceTest, ThrowsOnDuplicateName) {
    SearchSpace space;
    space.AddContinuous("a", 0.0, 1.0);
    EXPECT_THROW(space.AddInteger("a", 1, 2), std::invalid_argument);
}

TEST(SearchSpaceTest, ThrowsOnInvertedContinuousBounds) {
    SearchSpace space;
    EXPECT_THROW(space.AddContinuous("a", 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(space.AddContinuous("a", 1.0, 1.0), std::invalid_argument);
}

TEST(SearchSpaceTest, ThrowsOnNonPositiveLogUniformLower) {
    SearchSpace space;
    EXPECT_THROW(space.AddLogUniform("a", 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(space.AddLogUniform("a", -1.0, 1.0), std::invalid_argument);
}

TEST(SearchSpaceTest, ThrowsOnInvertedIntegerBounds) {
    SearchSpace space;
    EXPECT_THROW(space.AddInteger("a", 5, 4), std::invalid_argument);
}

TEST(SearchSpaceTest, IntegerBoundsEqualIsValid) {
    SearchSpace space;
    space.AddInteger("a", 5, 5);
    EXPECT_DOUBLE_EQ(space.Get("a").lower, 5.0);
}

TEST(SearchSpaceTest, ThrowsOnEmptyCategories) {
    SearchSpace space;
    EXPECT_THROW(space.AddCategorical("a", {}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
