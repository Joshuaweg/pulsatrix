/** @file neat_phenotype.hpp
 *  @brief Decodes a NEATGenome into an evaluable phenotype: a forward pass over the genome's
 *         own irregular connection graph.
 *  @ingroup evolutionary
 *  @note Deliberately NOT a Module subclass, and does not use ComputationGraph/Autograd at
 *        all -- per this campaign's own Risk Register (non-negotiable #5 has no established
 *        LRP literature for irregular/evolved topologies), a NEAT genome's arbitrary
 *        connection graph is explicitly excluded from Module/LRP compliance as a logged scope
 *        cut, not a silent gap. NEAT itself never needs backpropagation through this
 *        phenotype in the first place -- it trains via evolution (selection + mutation over
 *        the genome), not gradient descent, so no backward pass is genuinely needed here.
 *  @note Activation function: the original paper's own steepened sigmoid, 1/(1 + exp(-4.9x)),
 *        applied to every non-input node (Stanley & Miikkulainen 2002's own choice, cited
 *        directly rather than substituted for a different activation without comment).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

#include "pulsatrix/neat_genome.hpp"

namespace pulsatrix {

/**
 * @brief Evaluates genome's phenotype forward pass on inputs (one value per Input node,
 *        ordered by ascending node ID; the Bias node, if present, is always implicitly 1.0
 *        and is not part of inputs).
 * @return One value per Output node, ordered by ascending node ID.
 * @throws std::invalid_argument if inputs.size() doesn't match the genome's own number of
 *         Input nodes.
 * @note Recursive, memoized evaluation over the genome's connection graph -- safe against
 *       infinite recursion only because every genome constructed via NEATGenome's own public
 *       API (AddConnection's cycle check) is guaranteed acyclic; this function trusts that
 *       invariant rather than re-checking it, the same "pure core trusts its caller" division
 *       of responsibility used throughout this campaign.
 */
inline std::vector<double> EvaluateNEATPhenotype(const NEATGenome& genome, const std::vector<double>& inputs) {
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    int bias_id = -1;
    for (const auto& n : genome.nodes()) {
        switch (n.type) {
            case NodeGene::Type::Input:
                input_ids.push_back(n.id);
                break;
            case NodeGene::Type::Bias:
                bias_id = n.id;
                break;
            case NodeGene::Type::Output:
                output_ids.push_back(n.id);
                break;
            case NodeGene::Type::Hidden:
                break;
        }
    }
    std::sort(input_ids.begin(), input_ids.end());
    std::sort(output_ids.begin(), output_ids.end());

    if (inputs.size() != input_ids.size()) {
        throw std::invalid_argument("EvaluateNEATPhenotype: inputs.size() must match the genome's Input node count");
    }

    std::map<int, double> fixed_values;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        fixed_values[input_ids[i]] = inputs[i];
    }
    if (bias_id != -1) {
        fixed_values[bias_id] = 1.0;
    }

    std::map<int, double> cache;
    std::function<double(int)> evaluate = [&](int node_id) -> double {
        auto fixed_it = fixed_values.find(node_id);
        if (fixed_it != fixed_values.end()) {
            return fixed_it->second;
        }
        auto cached_it = cache.find(node_id);
        if (cached_it != cache.end()) {
            return cached_it->second;
        }
        double sum = 0.0;
        for (const auto& c : genome.connections()) {
            if (c.enabled && c.out_node == node_id) {
                sum += c.weight * evaluate(c.in_node);
            }
        }
        double activated = 1.0 / (1.0 + std::exp(-4.9 * sum));
        cache[node_id] = activated;
        return activated;
    };

    std::vector<double> outputs;
    outputs.reserve(output_ids.size());
    for (int id : output_ids) {
        outputs.push_back(evaluate(id));
    }
    return outputs;
}

}  // namespace pulsatrix
