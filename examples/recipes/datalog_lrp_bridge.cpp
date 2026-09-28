/** @file datalog_lrp_bridge.cpp
 *  @brief Recipe: a neural predicate's output wired into a Datalog derivation, then LRP
 *         relevance traced from the derived query fact back through the Datalog circuit and
 *         into the neural predicate's own raw input.
 */
#include <cstdio>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"

int main() {
    using namespace pulsatrix::datalog;
    using pulsatrix::CPUBackend;
    using pulsatrix::Shape;
    using pulsatrix::Tensor;

    CPUBackend backend;
    NeuralPredicateDatalogBridge bridge(&backend);

    // edge(a,b)'s weight comes from a real neural predicate (LinearModule(1,1) + sigmoid);
    // edge(a,c)=0.4, edge(b,d)=0.6, edge(c,d)=0.3 are constants. The program is:
    //   ancestor(X,Y) :- edge(X,Y).
    //   ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
    Tensor x(Shape({1, 1}), &backend, {0.5f});
    NeuralPredicateQueryResult query = bridge.evaluate(x);

    std::printf("Datalog LRP bridge recipe -- ancestor(a,d) via a neural-predicate-weighted edge\n\n");
    std::printf("ancestor(a,d) query weight:        %.6f\n", query.query_weight);
    std::printf("d(ancestor(a,d))/d(edge(a,b)):      %.6f\n\n", query.grad_wrt_predicate_output);

    bridge.backward();
    std::printf("predicate weight grad:              %.6f\n", bridge.predicate().weight_grad().at({0, 0}));
    std::printf("predicate bias grad:                %.6f\n\n", bridge.predicate().bias_grad().at({0}));

    // Seed relevance at the derived query fact and trace it back through the Datalog
    // derivation (edge/ancestor) and on into the predicate's own Module chain.
    constexpr double kSeed = 1.0;
    NeuralPredicateRelevanceResult relevance = bridge.propagate_relevance(kSeed);

    std::printf("Relevance seeded at ancestor(a,d) = %.2f, traced to every base fact:\n", kSeed);
    double total = 0.0;
    const auto edge_ab = Atom("edge", {Term::make_constant("a"), Term::make_constant("b")});
    for (const auto& [atom, r] : relevance.base_fact_relevance) {
        std::printf("  %s(%s, %s): %.6f\n", atom.predicate_name().c_str(), atom.terms()[0].value().c_str(),
                    atom.terms()[1].value().c_str(), r);
        if (atom != edge_ab) {
            total += r;
        }
    }
    const double r_x = static_cast<double>(relevance.relevance_wrt_x.at({0, 0}));
    std::printf("  predicate's raw input x (edge(a,b)'s relevance, continued through sigmoid +\n");
    std::printf("  LinearModule::propagate_relevance): %.6f\n\n", r_x);
    std::printf("conservation check: other base facts + relevance at x = %.6f (should equal seed %.2f)\n",
                total + r_x, kSeed);

    return 0;
}
