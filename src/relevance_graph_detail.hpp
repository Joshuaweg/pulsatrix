// Assembling an attribution graph from LRP at every layer boundary, shared by the language-model
// graph (relevance_graph.cpp, VIZ-4) and the residue graph (protein_explanations.cpp, PLM-6).
// Private to src/.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pulsatrix/relevance_graph.hpp"

namespace pulsatrix {
namespace relevance_graph_detail {

struct GraphInputs {
    int64_t positions = 0;
    int64_t hidden = 0;
    int64_t layers = 0;
    std::vector<int64_t> ids;
    /** @brief Relevance at each boundary, (1, positions, hidden) on the host: the embeddings,
     *         then each block's output. */
    std::vector<std::vector<float>> boundaries;
    /** @brief Block b's LRP from relevance at its output, (1, positions, hidden), to its input. */
    std::function<std::vector<float>(int64_t block, const std::vector<float>& relevance_out)> block_relevance;
    /** @brief The output node: where it sits, its feature id, its value and its label. */
    int64_t output_position = 0;
    int64_t output_feature = 0;
    double output_probability = 1.0;
    double output_value = 0.0;
    std::string output_label;
    std::string description;
};

/** @brief The graph: nodes per boundary and position, links per block and position, links from
 *         every top-boundary node with relevance to the output, then influence and pruning. */
AttributionGraph Assemble(const GraphInputs& in, const RelevanceGraphOptions& options);

}  // namespace relevance_graph_detail
}  // namespace pulsatrix
