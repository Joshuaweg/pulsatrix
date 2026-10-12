// INT-8: exporting embeddings for UMAP and t-SNE tools, and reading their coordinates back. The
// .npy files under tests/fixtures/npy were written by numpy (tools/generate_npy_fixtures.py):
// SerializeNpy() must reproduce them byte for byte, and ParseNpy() must read them.
#include "pulsatrix/embedding_export.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/svg.hpp"

namespace pulsatrix {
namespace {

std::vector<uint8_t> Fixture(const std::string& name) {
    std::ifstream in(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/npy/" + name, std::ios::binary);
    EXPECT_TRUE(in) << name;
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

class EmbeddingExportTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The fixtures' values: 0.25 k - 1 for the k-th value in C order.
    Tensor Values(const Shape& shape) {
        std::vector<float> v(static_cast<std::size_t>(shape.numel()));
        for (std::size_t k = 0; k < v.size(); ++k) v[k] = 0.25f * static_cast<float>(k) - 1.0f;
        return Tensor(shape, &backend, v);
    }

    static void ExpectSame(const Tensor& got, const Tensor& want) {
        ASSERT_EQ(got.shape(), want.shape());
        EXPECT_EQ(got.to_host_vector(), want.to_host_vector());
    }

    std::filesystem::path Scratch(const std::string& name) {
        const auto dir = std::filesystem::temp_directory_path() / ("pulsatrix_embedding_export_" + name);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        return dir;
    }
};

TEST_F(EmbeddingExportTest, SerializeNpyWritesWhatNumpyWrites) {
    EXPECT_EQ(SerializeNpy(Values(Shape({2, 3}))), Fixture("c_f4_2x3.npy"));
    EXPECT_EQ(SerializeNpy(Values(Shape({5}))), Fixture("vector_f4_5.npy"));
    EXPECT_EQ(SerializeNpy(Values(Shape({}))), Fixture("scalar_f4.npy"));
    EXPECT_EQ(SerializeNpy(Tensor(Shape({0, 3}), &backend)), Fixture("empty_f4_0x3.npy"));
}

TEST_F(EmbeddingExportTest, ParseNpyReadsWhatNumpyWrites) {
    ExpectSame(ParseNpy(Fixture("c_f4_2x3.npy"), &backend), Values(Shape({2, 3})));
    ExpectSame(ParseNpy(Fixture("c_f8_3x2.npy"), &backend), Values(Shape({3, 2})));
    // Fortran order on disk, C order in the tensor.
    ExpectSame(ParseNpy(Fixture("fortran_f4_2x3x4.npy"), &backend), Values(Shape({2, 3, 4})));
    ExpectSame(ParseNpy(Fixture("v2_f4_2x3.npy"), &backend), Values(Shape({2, 3})));
    ExpectSame(ParseNpy(Fixture("vector_f4_5.npy"), &backend), Values(Shape({5})));
    ExpectSame(ParseNpy(Fixture("scalar_f4.npy"), &backend), Values(Shape({})));
    EXPECT_EQ(ParseNpy(Fixture("empty_f4_0x3.npy"), &backend).shape(), Shape({0, 3}));
}

TEST_F(EmbeddingExportTest, ParseNpyRejectsWhatItCantRead) {
    auto rejects = [&](const std::vector<uint8_t>& bytes) { EXPECT_THROW((void)ParseNpy(bytes, &backend), std::invalid_argument); };
    rejects(Fixture("c_i8_2x2.npy"));          // integers
    rejects(Fixture("c_bigendian_f4_2.npy"));  // big-endian
    rejects({});
    rejects({'n', 'o', 't', ' ', 'n', 'p', 'y', '!', '!', '!'});

    const std::vector<uint8_t> good = Fixture("c_f4_2x3.npy");
    std::vector<uint8_t> truncated(good.begin(), good.end() - 1);
    rejects(truncated);
    std::vector<uint8_t> longer = good;
    longer.push_back(0);
    rejects(longer);
    std::vector<uint8_t> version = good;
    version[6] = 4;
    rejects(version);
    std::vector<uint8_t> header_too_long = good;
    header_too_long[8] = 0xff;
    header_too_long[9] = 0xff;
    rejects(header_too_long);

    // Headers that lie, or ask for more than the file holds.
    auto with_header = [](const std::string& dict, std::size_t data_bytes) {
        std::string h = dict;
        h.append((64 - (10 + h.size() + 1) % 64) % 64, ' ');
        h += '\n';
        std::vector<uint8_t> b = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, static_cast<uint8_t>(h.size()), 0};
        b.insert(b.end(), h.begin(), h.end());
        b.resize(b.size() + data_bytes, 0);
        return b;
    };
    EXPECT_NO_THROW((void)ParseNpy(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }", 8), &backend));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }", 8));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904, 4), }", 8));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904, 4, 0), }", 0));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (99999999999999999999,), }", 8));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 1, 1, 1, 1, 1, 1, 1), }", 4));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, }", 8));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), 'shape': (2,), }", 8));
    rejects(with_header("{'descr': '<f4', 'fortran_order': maybe, 'shape': (2,), }", 8));
    rejects(with_header("{'descr': '|O', 'fortran_order': False, 'shape': (2,), }", 16));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (2,), } junk", 8));
    rejects(with_header("{'descr': '<f4', 'fortran_order': False, 'shape': (-2,), }", 8));

    // A float64 beyond float32's range can't be read without losing it.
    std::vector<uint8_t> big = with_header("{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }", 0);
    const double huge = 1e300;
    uint64_t bits = 0;
    std::memcpy(&bits, &huge, 8);
    for (int k = 0; k < 8; ++k) big.push_back(static_cast<uint8_t>(bits >> (8 * k)));
    rejects(big);
}

TEST_F(EmbeddingExportTest, WriteNpyAndReadNpyRoundTrip) {
    const auto dir = Scratch("npy");
    const Tensor x = Values(Shape({4, 2, 3}));
    WriteNpy((dir / "x.npy").string(), x);
    ExpectSame(ReadNpy((dir / "x.npy").string(), &backend), x);
    EXPECT_THROW((void)ReadNpy((dir / "missing.npy").string(), &backend), std::runtime_error);
    EXPECT_THROW(WriteNpy((dir / "no_such_dir" / "x.npy").string(), x), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST_F(EmbeddingExportTest, ProjectorFilesFollowItsFormat) {
    // Nine significant digits read back as the same float32.
    const float third = 1.0f / 3.0f;
    const Tensor x(Shape({2, 3}), &backend, std::vector<float>{third, -2.5f, 1e-8f, 0.0f, 123456792.0f, -0.1f});
    const std::string tsv = ProjectorVectorsTsv(x);
    EXPECT_EQ(tsv, "0.333333343\t-2.5\t9.99999994e-09\n0\t123456792\t-0.100000001\n");
    EXPECT_EQ(std::stof("0.333333343"), third);
    // A convolution's (n, c, h, w) activations are n points.
    EXPECT_EQ(ProjectorVectorsTsv(Values(Shape({2, 1, 2}))), "-1\t-0.75\n-0.5\t-0.25\n");
    EXPECT_THROW((void)ProjectorVectorsTsv(Values(Shape({3}))), std::invalid_argument);
    EXPECT_THROW((void)ProjectorVectorsTsv(Tensor(Shape({1, 2}), &backend, std::vector<float>{1.0f, std::nanf("")})), std::invalid_argument);

    // One column: no header, or the projector would take it for a point. Two: a header.
    EXPECT_EQ(ProjectorMetadataTsv({{"label", {"cat", "dog"}}}), "cat\ndog\n");
    EXPECT_EQ(ProjectorMetadataTsv({{"label", {"cat", "a\tb\nc"}}, {"class", {"0", "1"}}}), "label\tclass\ncat\t0\na b c\t1\n");
    EXPECT_THROW((void)ProjectorMetadataTsv({}), std::invalid_argument);
    EXPECT_THROW((void)ProjectorMetadataTsv({{"label", {"cat", "dog"}}, {"class", {"0"}}}), std::invalid_argument);
}

TEST_F(EmbeddingExportTest, ExportEmbeddingsWritesEveryFormat) {
    const auto dir = Scratch("export");
    const Tensor x = Values(Shape({3, 2}));
    ExportEmbeddings(dir.string(), x, {{"label", {"a", "b", "c"}}, {"class", {"x", "y", "x"}}});
    ExpectSame(ReadNpy((dir / "vectors.npy").string(), &backend), x);
    EXPECT_EQ(ReadText(dir / "vectors.tsv"), ProjectorVectorsTsv(x));
    EXPECT_EQ(ReadText(dir / "metadata.tsv"), "label\tclass\na\tx\nb\ty\nc\tx\n");

    const auto bare = Scratch("bare");
    ExportEmbeddings(bare.string(), x);
    EXPECT_TRUE(std::filesystem::exists(bare / "vectors.npy"));
    EXPECT_FALSE(std::filesystem::exists(bare / "metadata.tsv"));
    EXPECT_THROW(ExportEmbeddings(bare.string(), x, {{"label", {"a", "b"}}}), std::invalid_argument);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(bare);
}

TEST_F(EmbeddingExportTest, CoordinatesComeBackAsAProjectionThatSaysWhatItDistorts) {
    // As if UMAP had written its 2D output with np.save.
    const auto dir = Scratch("umap");
    WriteNpy((dir / "umap.npy").string(), Values(Shape({4, 2})));
    const Tensor coordinates = ReadNpy((dir / "umap.npy").string(), &backend);
    std::filesystem::remove_all(dir);

    const ProjectionDocument doc = MakeProjectionDocument(coordinates, "umap", {"a", "b", "c", "d"}, {"x", "x", "y", "y"}, {}, "Layer 6, UMAP");
    EXPECT_EQ(doc.axes, (std::vector<std::string>{"umap 1", "umap 2"}));
    EXPECT_TRUE(doc.explained_variance_ratio.empty());
    EXPECT_EQ(doc.coordinates, coordinates.to_host_vector());
    EXPECT_TRUE(IsNeighborEmbedding(doc));
    const ProjectionDocument back = ParseProjectionDocument(ToJson(doc));
    EXPECT_EQ(back.method, "umap");
    EXPECT_EQ(back.coordinates, doc.coordinates);

    const std::string note = "distances, cluster sizes and densities";
    EXPECT_NE(RenderProjectionSvg(doc).find(note), std::string::npos);
    EXPECT_NE(RenderProjectionHtml(doc).find(note), std::string::npos);
    ProjectionDocument linear = doc;
    linear.method = "pca";
    EXPECT_FALSE(IsNeighborEmbedding(linear));
    EXPECT_EQ(RenderProjectionSvg(linear).find(note), std::string::npos);
    EXPECT_EQ(RenderProjectionHtml(linear).find(note), std::string::npos);
    for (const char* m : {"UMAP", "tsne", "t-SNE", "openTSNE"}) {
        linear.method = m;
        EXPECT_TRUE(IsNeighborEmbedding(linear)) << m;
    }

    EXPECT_THROW((void)MakeProjectionDocument(Values(Shape({4, 1})), "umap"), std::invalid_argument);
    EXPECT_THROW((void)MakeProjectionDocument(Values(Shape({4, 2})), ""), std::invalid_argument);
    EXPECT_THROW((void)MakeProjectionDocument(Values(Shape({4, 2})), "umap", {"too", "few"}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
