/** @file circuit_graph_view.hpp
 *  @brief Node-link diagram for a CircuitGraph (mechanistic-interpretability visualization).
 *  @ingroup visualization
 */
#pragma once

#include "pulsatrix/circuit_graph.hpp"
#include "pulsatrix/viz/document.hpp"

namespace pulsatrix {

/**
 * @brief Draws a CircuitGraph as a node-link diagram: node position is topological depth
 *        (x-axis), node size and color encode ablation_effect via length/area AND Viridis
 *        (unsigned magnitude -- never hue-for-magnitude, hc_information_visualization.md
 *        SS1), edge thickness encodes weight. Each node is captioned with
 *        CircuitNodeDisplayLabel (its own label, else op type + id) above and its ablation
 *        effect below.
 * @note v1 scope: every ComputationGraph this codebase builds today is a single-parent
 *       linear chain (see circuit_graph.hpp's own note on CircuitEdge::weight), so a single
 *       horizontal row of nodes is a faithful layout. A general DAG force-layout is
 *       deferred until a genuinely branching architecture exists to visualize.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note.
 */
class CircuitGraphView {
public:
    static void Draw(const char* title, const CircuitGraph& graph);

    /** @brief Draws a `pulsatrix.circuit_graph.v1` document the same way. */
    static void Draw(const char* title, const CircuitGraphDocument& doc);
};

}  // namespace pulsatrix
