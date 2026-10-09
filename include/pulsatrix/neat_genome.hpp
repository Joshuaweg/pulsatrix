/** @file neat_genome.hpp
 *  @brief NEAT genome (Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting
 *         Topologies," Evolutionary Computation 10(2), 2002): a connection-gene list with
 *         global historical markings (innovation numbers) plus the two structural mutations
 *         (add-connection, add-node) that grow topology from a minimal starting point.
 *  @ingroup evolutionary
 *  @note Feedforward-only, logged as a deliberate scope reduction matching this project's own
 *        standing precedent (the Reinforcement Learning campaign's own "feedforward-only for
 *        now" scope note) -- MutateAddConnection's RNG-driven wrapper only ever proposes a
 *        connection that cannot create a cycle (checked via reachability, not merely assumed
 *        safe), rather than NEAT's own original recurrent-connection-capable design.
 *  @note Phenotype decode (turning a genome into an evaluable Module-based network) is
 *        deliberately NOT this file's job -- this mission's own scope is the genome and its
 *        structural mutations only; decode is Phase 3 Mission 1's job.
 */
#pragma once

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsatrix {

/** @brief One node in a NEAT genome's topology. */
struct NodeGene {
    enum class Type { Input, Bias, Output, Hidden };
    int id;
    Type type;
};

/** @brief One connection in a NEAT genome: an edge between two node IDs, its weight, whether
 *         it is currently active, and its historical marking (innovation number). Disabled
 *         connections are kept, not removed -- NEAT's own design, preserving historical
 *         alignment for crossover (a future mission's concern, not built here). */
struct ConnectionGene {
    int in_node;
    int out_node;
    double weight;
    bool enabled;
    int innovation;
};

/**
 * @brief The global historical-marking registry: the same structural mutation (an identical
 *        new connection, or an identical connection-split creating a new node) occurring in
 *        different genomes receives the *same* innovation number / new node ID if it has
 *        already been recorded, and a fresh one otherwise. This is the concrete mechanism
 *        that lets two differently-shaped genomes' genes be meaningfully aligned by innovation
 *        number -- NEAT's own defining idea (not built here: crossover itself is a future
 *        mission; this class only maintains the registry crossover would eventually consume).
 */
class InnovationTracker {
public:
    explicit InnovationTracker(int next_node_id) : next_node_id_(next_node_id) {}

    /** @brief The innovation number for a new (in_node, out_node) connection -- reused if
     *         this exact connection has already been recorded by any prior call, on this or
     *         any other genome sharing this tracker. */
    int GetConnectionInnovation(int in_node, int out_node) {
        auto key = std::make_pair(in_node, out_node);
        auto it = connection_innovations_.find(key);
        if (it != connection_innovations_.end()) {
            return it->second;
        }
        int innovation = next_innovation_++;
        connection_innovations_[key] = innovation;
        return innovation;
    }

    /** @brief The new node ID created by splitting the connection identified by
     *         split_connection_innovation -- reused if this exact connection has already been
     *         split by any prior call. */
    int GetNodeIdForSplit(int split_connection_innovation) {
        auto it = node_ids_by_split_.find(split_connection_innovation);
        if (it != node_ids_by_split_.end()) {
            return it->second;
        }
        int node_id = next_node_id_++;
        node_ids_by_split_[split_connection_innovation] = node_id;
        return node_id;
    }

private:
    std::map<std::pair<int, int>, int> connection_innovations_;
    std::map<int, int> node_ids_by_split_;
    int next_innovation_ = 0;
    int next_node_id_;
};

/** @brief A NEAT genome: its node and connection genes, growable via structural mutation. */
class NEATGenome {
public:
    /**
     * @brief Constructs the minimal starting topology: num_inputs input nodes (+1 bias node
     *        if has_bias), num_outputs output nodes, every input(+bias) node directly
     *        connected to every output node -- NEAT's own "complexify from minimal structure"
     *        starting point, not a large fixed topology later pruned.
     * @throws std::invalid_argument if num_inputs or num_outputs is 0.
     */
    NEATGenome(int num_inputs, int num_outputs, bool has_bias, InnovationTracker& tracker) {
        if (num_inputs <= 0 || num_outputs <= 0) {
            throw std::invalid_argument("NEATGenome: num_inputs and num_outputs must be positive");
        }
        int next_id = 0;
        std::vector<int> input_ids;
        for (int i = 0; i < num_inputs; ++i) {
            nodes_.push_back(NodeGene{next_id, NodeGene::Type::Input});
            input_ids.push_back(next_id);
            ++next_id;
        }
        if (has_bias) {
            nodes_.push_back(NodeGene{next_id, NodeGene::Type::Bias});
            input_ids.push_back(next_id);
            ++next_id;
        }
        std::vector<int> output_ids;
        for (int i = 0; i < num_outputs; ++i) {
            nodes_.push_back(NodeGene{next_id, NodeGene::Type::Output});
            output_ids.push_back(next_id);
            ++next_id;
        }
        for (int in_id : input_ids) {
            for (int out_id : output_ids) {
                int innovation = tracker.GetConnectionInnovation(in_id, out_id);
                connections_.push_back(ConnectionGene{in_id, out_id, 0.0, true, innovation});
            }
        }
    }

    [[nodiscard]] const std::vector<NodeGene>& nodes() const { return nodes_; }
    [[nodiscard]] const std::vector<ConnectionGene>& connections() const { return connections_; }

    /**
     * @brief Directly sets an existing connection's weight by innovation number -- needed
     *        infrastructure found necessary by Phase 3 Mission 1 (deterministic phenotype-
     *        evaluation tests need an exact, controllable weight; a future crossover mission
     *        will need the same capability to copy weights between genomes), logged here as
     *        a small scope addition beyond Mission 0's own original scope.
     * @throws std::invalid_argument if no connection with that innovation number exists.
     */
    void SetConnectionWeight(int innovation, double weight) {
        auto it = std::find_if(connections_.begin(), connections_.end(),
                                [&](const ConnectionGene& c) { return c.innovation == innovation; });
        if (it == connections_.end()) {
            throw std::invalid_argument("NEATGenome::SetConnectionWeight: no connection with that innovation");
        }
        it->weight = weight;
    }

    /**
     * @brief Pure core: adds a new enabled connection gene (in_node, out_node, weight),
     *        assigning its innovation number via tracker (reused if this exact connection has
     *        already been created on any genome sharing the tracker).
     * @throws std::invalid_argument if in_node or out_node doesn't exist in this genome, or a
     *         connection between them (in either direction) already exists.
     */
    void AddConnectionBetween(int in_node, int out_node, double weight, InnovationTracker& tracker) {
        if (!HasNode(in_node) || !HasNode(out_node)) {
            throw std::invalid_argument("NEATGenome::AddConnectionBetween: node not found in this genome");
        }
        for (const auto& c : connections_) {
            if (c.in_node == in_node && c.out_node == out_node) {
                throw std::invalid_argument("NEATGenome::AddConnectionBetween: connection already exists");
            }
        }
        int innovation = tracker.GetConnectionInnovation(in_node, out_node);
        connections_.push_back(ConnectionGene{in_node, out_node, weight, true, innovation});
    }

    /**
     * @brief RNG-driven wrapper: proposes a random, currently-nonexistent, feedforward-safe
     *        (cannot create a cycle -- checked via reachability, not merely assumed) pair of
     *        nodes and adds the connection with a small random initial weight. A no-op
     *        (returns false) if no such pair exists (e.g. the genome is already fully
     *        connected in every safe direction).
     */
    template <typename RNG>
    bool AddConnection(InnovationTracker& tracker, RNG& rng) {
        std::vector<std::pair<int, int>> candidates;
        for (const auto& a : nodes_) {
            if (a.type == NodeGene::Type::Output) {
                continue;  // outputs never originate a connection in a feedforward network
            }
            for (const auto& b : nodes_) {
                if (b.type == NodeGene::Type::Input || b.type == NodeGene::Type::Bias || a.id == b.id) {
                    continue;
                }
                if (ConnectionExists(a.id, b.id)) {
                    continue;
                }
                if (CanReach(b.id, a.id)) {
                    continue;  // adding a.id -> b.id would close a cycle
                }
                candidates.emplace_back(a.id, b.id);
            }
        }
        if (candidates.empty()) {
            return false;
        }
        std::uniform_int_distribution<size_t> pick(0, candidates.size() - 1);
        std::normal_distribution<double> weight_dist(0.0, 1.0);
        auto [in_node, out_node] = candidates[pick(rng)];
        AddConnectionBetween(in_node, out_node, weight_dist(rng), tracker);
        return true;
    }

    /**
     * @brief Pure core: splits the enabled connection with the given innovation number --
     *        disables it (kept, not removed), adds a new hidden node, and adds two new
     *        connections: in_node -> new_node (weight 1.0) and new_node -> out_node (weight =
     *        the disabled connection's own original weight). Per the original paper's own
     *        convention: this specific weight choice keeps the network's immediate behavior
     *        close to what it was right before the mutation, minimizing initial disruption.
     * @throws std::invalid_argument if no *enabled* connection with that innovation number
     *         exists in this genome.
     */
    void AddNodeSplitting(int connection_innovation, InnovationTracker& tracker) {
        auto it = std::find_if(connections_.begin(), connections_.end(), [&](const ConnectionGene& c) {
            return c.innovation == connection_innovation && c.enabled;
        });
        if (it == connections_.end()) {
            throw std::invalid_argument("NEATGenome::AddNodeSplitting: no enabled connection with that innovation");
        }
        it->enabled = false;
        int in_node = it->in_node;
        int out_node = it->out_node;
        double original_weight = it->weight;

        int new_node_id = tracker.GetNodeIdForSplit(connection_innovation);
        nodes_.push_back(NodeGene{new_node_id, NodeGene::Type::Hidden});

        int innovation_in = tracker.GetConnectionInnovation(in_node, new_node_id);
        connections_.push_back(ConnectionGene{in_node, new_node_id, 1.0, true, innovation_in});
        int innovation_out = tracker.GetConnectionInnovation(new_node_id, out_node);
        connections_.push_back(ConnectionGene{new_node_id, out_node, original_weight, true, innovation_out});
    }

    /**
     * @brief RNG-driven wrapper: splits a uniformly-randomly chosen enabled connection. A
     *        no-op (returns false) if no enabled connection exists.
     */
    template <typename RNG>
    bool AddNode(InnovationTracker& tracker, RNG& rng) {
        std::vector<int> enabled_innovations;
        for (const auto& c : connections_) {
            if (c.enabled) {
                enabled_innovations.push_back(c.innovation);
            }
        }
        if (enabled_innovations.empty()) {
            return false;
        }
        std::uniform_int_distribution<size_t> pick(0, enabled_innovations.size() - 1);
        AddNodeSplitting(enabled_innovations[pick(rng)], tracker);
        return true;
    }

    /**
     * @brief Non-structural mutation: perturbs every enabled connection's weight
     *        independently with probability mutation_probability by adding N(0, sigma^2)
     *        noise -- not part of NEAT's own named "structural mutation" pair, but genuinely
     *        necessary infrastructure for any real evolutionary run (a genome that only ever
     *        grows topology, never adjusts weights, cannot meaningfully learn); added here as
     *        a small, logged scope extension beyond this mission's literal title.
     * @throws std::invalid_argument if sigma < 0 or mutation_probability is outside [0, 1].
     */
    template <typename RNG>
    void MutateWeights(double sigma, double mutation_probability, RNG& rng) {
        if (sigma < 0.0) {
            throw std::invalid_argument("NEATGenome::MutateWeights: sigma must be non-negative");
        }
        if (mutation_probability < 0.0 || mutation_probability > 1.0) {
            throw std::invalid_argument("NEATGenome::MutateWeights: mutation_probability must be in [0, 1]");
        }
        std::bernoulli_distribution mask(mutation_probability);
        // Scaled standard normal draws: std::normal_distribution requires a positive stddev.
        std::normal_distribution<double> noise(0.0, 1.0);
        for (auto& c : connections_) {
            if (c.enabled && mask(rng)) {
                c.weight += noise(rng) * sigma;
            }
        }
    }

private:
    [[nodiscard]] bool HasNode(int id) const {
        return std::any_of(nodes_.begin(), nodes_.end(), [&](const NodeGene& n) { return n.id == id; });
    }

    [[nodiscard]] bool ConnectionExists(int in_node, int out_node) const {
        return std::any_of(connections_.begin(), connections_.end(), [&](const ConnectionGene& c) {
            return c.in_node == in_node && c.out_node == out_node;
        });
    }

    /** @brief Whether out_node is reachable from from_node following enabled connections --
     *         used to reject a candidate new connection that would close a cycle. */
    [[nodiscard]] bool CanReach(int from_node, int out_node) const {
        std::vector<int> stack{from_node};
        std::vector<int> visited;
        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            if (current == out_node) {
                return true;
            }
            if (std::find(visited.begin(), visited.end(), current) != visited.end()) {
                continue;
            }
            visited.push_back(current);
            for (const auto& c : connections_) {
                if (c.enabled && c.in_node == current) {
                    stack.push_back(c.out_node);
                }
            }
        }
        return false;
    }

    std::vector<NodeGene> nodes_;
    std::vector<ConnectionGene> connections_;
};

}  // namespace pulsatrix
