// TDA-4: Vietoris-Rips persistent homology, checked against results that don't depend on any
// implementation: the homotopy types of evenly spaced points on a circle (Adamaszek and Adams,
// Pacific J. Math. 2017), the Euler characteristic at every scale, the cocycle condition, and H0
// from the minimum spanning tree.
#include "pulsatrix/tda/rips_persistence.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tda/h0.hpp"

namespace pulsatrix {
namespace {

constexpr double kPi = 3.14159265358979323846;

// n evenly spaced points on the unit circle, with every chord of k steps given exactly the same
// float, so the theorem's ties are ties here too.
DistanceMatrix CirclePoints(int64_t n) {
    DistanceMatrix m;
    m.size = n;
    m.values.resize(static_cast<std::size_t>(n * (n - 1) / 2));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            const int64_t steps = std::min(j - i, n - (j - i));
            const float chord = static_cast<float>(2 * std::sin(kPi * static_cast<double>(steps) / static_cast<double>(n)));
            m.values[m.index(i, j)] = chord;
            m.max_distance = std::max(m.max_distance, chord);
        }
    }
    return m;
}

double Chord(int64_t steps, int64_t n) {
    return static_cast<float>(2 * std::sin(kPi * static_cast<double>(steps) / static_cast<double>(n)));
}

class RipsPersistenceTest : public ::testing::Test {
protected:
    CPUBackend backend;

    DistanceMatrix Random(int64_t n, int64_t d, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal;
        std::vector<float> v(static_cast<std::size_t>(n * d));
        for (float& x : v) x = normal(rng);
        return PairwiseDistances(Tensor(Shape({n, d}), &backend, v), {DistanceMethod::Exact});
    }

    // Points on a circle of radius r around (cx, cy), with a little noise.
    static void AddCircle(std::vector<float>& v, int count, float r, float cx, float cy, std::mt19937& rng) {
        std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
        for (int k = 0; k < count; ++k) {
            const double a = 2 * kPi * k / count;
            v.push_back(cx + r * static_cast<float>(std::cos(a)) + jitter(rng));
            v.push_back(cy + r * static_cast<float>(std::sin(a)) + jitter(rng));
        }
    }
};

TEST_F(RipsPersistenceTest, TwelvePointsOnACircleHaveTheirKnownLoopAndThreeVoids) {
    // Adamaszek and Adams: joining points within k of 12 steps gives a circle for k < 4, a wedge of
    // 12 - 2*4 - 1 = 3 two-spheres at k = 4, and a 5-sphere at k = 5.
    RipsOptions o;
    o.max_dimension = 2;
    const RipsResult r = ComputeRipsPersistence(CirclePoints(12), o);
    ASSERT_EQ(r.diagrams.size(), 3u);
    EXPECT_DOUBLE_EQ(r.threshold, Chord(6, 12));  // every point's farthest is the diameter, 2

    ASSERT_EQ(r.diagrams[1].pairs.size(), 1u);
    EXPECT_DOUBLE_EQ(r.diagrams[1].pairs[0].birth, Chord(1, 12));
    EXPECT_DOUBLE_EQ(r.diagrams[1].pairs[0].death, Chord(4, 12));

    ASSERT_EQ(r.diagrams[2].pairs.size(), 3u);
    for (const PersistencePair& pair : r.diagrams[2].pairs) {
        EXPECT_DOUBLE_EQ(pair.birth, Chord(4, 12));
        EXPECT_DOUBLE_EQ(pair.death, Chord(5, 12));
    }

    // H0: twelve points join at the side length, leaving one component.
    ASSERT_EQ(r.diagrams[0].pairs.size(), 12u);
    EXPECT_DOUBLE_EQ(r.diagrams[0].pairs[10].death, Chord(1, 12));
    EXPECT_TRUE(r.diagrams[0].pairs[11].essential());
}

TEST_F(RipsPersistenceTest, TheCircleGivesTheSameDiagramsOverZ3AndZ5) {
    RipsOptions o;
    o.max_dimension = 2;
    const RipsResult z2 = ComputeRipsPersistence(CirclePoints(12), o);
    for (int p : {3, 5}) {
        o.coefficient = p;
        const RipsResult zp = ComputeRipsPersistence(CirclePoints(12), o);
        for (std::size_t d = 0; d < 3; ++d) {
            ASSERT_EQ(zp.diagrams[d].pairs.size(), z2.diagrams[d].pairs.size()) << "p " << p << " dim " << d;
            for (std::size_t k = 0; k < zp.diagrams[d].pairs.size(); ++k) {
                EXPECT_EQ(zp.diagrams[d].pairs[k].birth, z2.diagrams[d].pairs[k].birth);
                EXPECT_EQ(zp.diagrams[d].pairs[k].death, z2.diagrams[d].pairs[k].death);
            }
        }
    }
}

TEST_F(RipsPersistenceTest, TheEulerCharacteristicMatchesAtEveryScale) {
    // With every dimension computed, sum (-1)^d Betti_d at scale t equals sum (-1)^d (number of
    // d-simplices of diameter <= t). Betti_d at t counts pairs with birth <= t < death.
    for (uint32_t seed : {1u, 2u, 3u}) {
        const int64_t n = 8;
        const DistanceMatrix m = Random(n, 3, seed);
        RipsOptions o;
        o.max_dimension = static_cast<int>(n) - 2;
        o.threshold = m.max_distance;
        for (int p : {2, 3}) {
            o.coefficient = p;
            const RipsResult r = ComputeRipsPersistence(m, o);
            std::vector<double> scales(m.values.begin(), m.values.end());
            for (double t : scales) {
                int64_t betti_sum = 0;
                for (const PersistenceDiagram& d : r.diagrams) {
                    for (const PersistencePair& pair : d.pairs) {
                        if (pair.birth <= t && t < pair.death) betti_sum += (d.dimension % 2 == 0) ? 1 : -1;
                    }
                }
                // Every nonempty subset of the points is a simplex once t reaches its diameter.
                // Dimensions 0 to n - 2 are all the homology there is: the one (n - 1)-simplex
                // can't be a cycle.
                int64_t simplex_sum = 0;
                for (int mask = 1; mask < (1 << n); ++mask) {
                    std::vector<int64_t> pts;
                    for (int64_t i = 0; i < n; ++i) {
                        if (mask & (1 << i)) pts.push_back(i);
                    }
                    float diameter = 0;
                    for (std::size_t a = 0; a < pts.size(); ++a) {
                        for (std::size_t b = a + 1; b < pts.size(); ++b) diameter = std::max(diameter, m.at(pts[a], pts[b]));
                    }
                    if (static_cast<double>(diameter) <= t) simplex_sum += (pts.size() % 2 == 1) ? 1 : -1;
                }
                EXPECT_EQ(betti_sum, simplex_sum) << "seed " << seed << " p " << p << " t " << t;
            }
        }
    }
}

// The rank of a matrix over Z/p, by Gaussian elimination.
int64_t RankModP(std::vector<std::vector<int>> m, int p) {
    int64_t rank = 0;
    const std::size_t rows = m.size(), cols = rows == 0 ? 0 : m[0].size();
    for (std::size_t c = 0; c < cols && rank < static_cast<int64_t>(rows); ++c) {
        std::size_t r = static_cast<std::size_t>(rank);
        while (r < rows && m[r][c] == 0) ++r;
        if (r == rows) continue;
        std::swap(m[r], m[static_cast<std::size_t>(rank)]);
        std::vector<int>& pivot = m[static_cast<std::size_t>(rank)];
        int inv = 1;
        while ((pivot[c] * inv) % p != 1) ++inv;
        for (int& x : pivot) x = (x * inv) % p;
        for (std::size_t q = 0; q < rows; ++q) {
            if (q == static_cast<std::size_t>(rank) || m[q][c] == 0) continue;
            const int f = m[q][c];
            for (std::size_t k = 0; k < cols; ++k) m[q][k] = ((m[q][k] - f * pivot[k]) % p + p) % p;
        }
        ++rank;
    }
    return rank;
}

TEST_F(RipsPersistenceTest, BettiNumbersMatchBoundaryRanksAtEveryScale) {
    // Betti_d = #d-simplices - rank(boundary_d) - rank(boundary_{d+1}), computed from scratch for
    // every subset of 7 points at every scale, against the pairs alive at that scale.
    const int64_t n = 7;
    for (uint32_t seed : {4u, 5u}) {
        const DistanceMatrix m = Random(n, 2, seed);
        for (int p : {2, 3}) {
            RipsOptions o;
            o.max_dimension = static_cast<int>(n) - 2;
            o.threshold = m.max_distance;
            o.coefficient = p;
            const RipsResult r = ComputeRipsPersistence(m, o);
            for (float scale : m.values) {
                const double t = scale;
                std::vector<std::vector<int>> by_dim(static_cast<std::size_t>(n));  // masks per dimension
                for (int mask = 1; mask < (1 << n); ++mask) {
                    float diameter = 0;
                    int size = 0;
                    for (int64_t i = 0; i < n; ++i) {
                        if (!(mask & (1 << i))) continue;
                        ++size;
                        for (int64_t j = i + 1; j < n; ++j) {
                            if (mask & (1 << j)) diameter = std::max(diameter, m.at(i, j));
                        }
                    }
                    if (static_cast<double>(diameter) <= t) by_dim[static_cast<std::size_t>(size - 1)].push_back(mask);
                }
                // rank of the boundary from dimension d to d - 1, with signs (-1)^position.
                auto boundary_rank = [&](int d) -> int64_t {
                    if (d <= 0 || d >= static_cast<int>(n)) return 0;
                    const auto& rows = by_dim[static_cast<std::size_t>(d - 1)];
                    const auto& cols = by_dim[static_cast<std::size_t>(d)];
                    if (rows.empty() || cols.empty()) return 0;
                    std::vector<std::vector<int>> b(rows.size(), std::vector<int>(cols.size(), 0));
                    for (std::size_t c = 0; c < cols.size(); ++c) {
                        int position = 0;
                        for (int64_t i = 0; i < n; ++i) {
                            if (!(cols[c] & (1 << i))) continue;
                            const int face = cols[c] & ~(1 << i);
                            const auto at = std::find(rows.begin(), rows.end(), face);
                            b[static_cast<std::size_t>(at - rows.begin())][c] = (position % 2 == 0) ? 1 : p - 1;
                            ++position;
                        }
                    }
                    return RankModP(b, p);
                };
                for (int d = 0; d <= o.max_dimension; ++d) {
                    const int64_t betti = static_cast<int64_t>(by_dim[static_cast<std::size_t>(d)].size()) -
                                          boundary_rank(d) - boundary_rank(d + 1);
                    int64_t alive = 0;
                    for (const PersistencePair& pair : r.diagrams[static_cast<std::size_t>(d)].pairs) {
                        if (pair.birth <= t && t < pair.death) ++alive;
                    }
                    EXPECT_EQ(alive, betti) << "seed " << seed << " p " << p << " dim " << d << " t " << t;
                }
            }
        }
    }
}

TEST_F(RipsPersistenceTest, HZeroMatchesTheMinimumSpanningTree) {
    const DistanceMatrix m = Random(60, 4, 7);
    const RipsResult r = ComputeRipsPersistence(m, {});
    const H0Result mst = ComputeH0(m);
    ASSERT_EQ(r.diagrams[0].pairs.size(), mst.diagram.pairs.size());
    for (std::size_t k = 0; k < mst.diagram.pairs.size(); ++k) {
        EXPECT_EQ(r.diagrams[0].pairs[k].death, mst.diagram.pairs[k].death);
    }
    for (const RipsFeature& f : r.features[0]) {
        if (f.pair.essential()) continue;
        EXPECT_EQ(m.at(f.death_edge[0], f.death_edge[1]), static_cast<float>(f.pair.death));
    }
}

TEST_F(RipsPersistenceTest, TwoNoisyCirclesGiveTwoLongLoops) {
    std::mt19937 rng(3);
    std::vector<float> v;
    AddCircle(v, 30, 1.0f, 0.0f, 0.0f, rng);
    AddCircle(v, 24, 0.6f, 4.0f, 0.0f, rng);
    const DistanceMatrix m = PairwiseDistances(Tensor(Shape({54, 2}), &backend, v));
    const RipsResult r = ComputeRipsPersistence(m, {});
    std::vector<double> long_lived;
    for (const PersistencePair& pair : r.diagrams[1].pairs) {
        if (pair.persistence() > 0.5) long_lived.push_back(pair.persistence());
    }
    EXPECT_EQ(long_lived.size(), 2u);
    // Every feature's critical edges have its birth and death as their lengths.
    for (const RipsFeature& f : r.features[1]) {
        EXPECT_EQ(m.at(f.birth_edge[0], f.birth_edge[1]), static_cast<float>(f.pair.birth));
        ASSERT_FALSE(f.pair.essential());
        EXPECT_EQ(m.at(f.death_edge[0], f.death_edge[1]), static_cast<float>(f.pair.death));
    }
    EXPECT_EQ(r.diagrams[1].error_bound, m.error_bound);
}

TEST_F(RipsPersistenceTest, CocyclesAreCocyclesAndWindAroundTheCircle) {
    for (int p : {2, 7}) {
        const int64_t n = 16;
        RipsOptions o;
        o.cocycles = true;
        o.coefficient = p;
        const DistanceMatrix m = CirclePoints(n);
        const RipsResult r = ComputeRipsPersistence(m, o);
        ASSERT_EQ(r.features[1].size(), 1u);
        const RipsFeature& loop = r.features[1][0];
        ASSERT_FALSE(loop.cocycle.empty());

        // phi on an edge (a, b) with a < b; phi(b, a) = -phi(a, b).
        auto phi = [&](int64_t a, int64_t b) {
            const int sign = a < b ? 1 : -1;
            for (const CocycleEntry& e : loop.cocycle) {
                if (e.vertices[0] == std::min(a, b) && e.vertices[1] == std::max(a, b)) {
                    return ((sign * e.coefficient) % p + p) % p;
                }
            }
            return 0;
        };
        // Cocycle condition on every triangle present before the death:
        // phi(b, c) - phi(a, c) + phi(a, b) = 0 mod p.
        for (int64_t a = 0; a < n; ++a) {
            for (int64_t b = a + 1; b < n; ++b) {
                for (int64_t c = b + 1; c < n; ++c) {
                    const double diameter = std::max({m.at(a, b), m.at(a, c), m.at(b, c)});
                    if (diameter >= loop.pair.death) continue;
                    EXPECT_EQ(((phi(b, c) - phi(a, c) + phi(a, b)) % p + p) % p, 0) << a << b << c;
                }
            }
        }
        // Around the circle, step by step, it adds up to a nonzero class.
        int winding = 0;
        for (int64_t a = 0; a < n; ++a) winding = (winding + phi(a, (a + 1) % n)) % p;
        EXPECT_NE(winding, 0) << "p " << p;
    }
}

TEST_F(RipsPersistenceTest, ALowThresholdLeavesLoopsAliveAsEssential) {
    RipsOptions o;
    o.threshold = Chord(2, 12);  // the loop is born at chord 1 and would die at chord 4
    const RipsResult r = ComputeRipsPersistence(CirclePoints(12), o);
    ASSERT_EQ(r.diagrams[1].pairs.size(), 1u);
    EXPECT_TRUE(r.diagrams[1].pairs[0].essential());
    EXPECT_EQ(r.features[1][0].death_edge[0], -1);
    EXPECT_EQ(r.diagrams[1].essential_count(), 1);
}

TEST_F(RipsPersistenceTest, BadOptionsThrow) {
    const DistanceMatrix m = CirclePoints(6);
    RipsOptions o;
    o.coefficient = 4;
    EXPECT_THROW((void)ComputeRipsPersistence(m, o), std::invalid_argument);
    o.coefficient = 1;
    EXPECT_THROW((void)ComputeRipsPersistence(m, o), std::invalid_argument);
    o = {};
    o.max_dimension = 9;
    EXPECT_THROW((void)ComputeRipsPersistence(m, o), std::invalid_argument);
    o.max_dimension = -1;
    EXPECT_THROW((void)ComputeRipsPersistence(m, o), std::invalid_argument);
    DistanceMatrix one;
    one.size = 1;
    EXPECT_THROW((void)ComputeRipsPersistence(one, {}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
