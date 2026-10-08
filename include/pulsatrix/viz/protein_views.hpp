/** @file protein_views.hpp
 *  @brief Figures and pages for proteins (PLM-5): the mutation map, the sequence logo, the
 *         contact map, residue tracks, and a 3D structure colored by any per-residue score.
 *  @ingroup visualization
 *
 *  The SVG figures follow svg.hpp's rules: standalone files, deterministic bytes, no font library,
 *  and the same colors as every other view (blue below zero, white at zero, red above; Viridis
 *  for magnitudes). Long sequences wrap into blocks of positions, numbered as the document
 *  numbers them. Each cell or column carries a tooltip (`<title>`).
 *
 *  The HTML pages hold the same figure inline, with no scripts, so they work offline and in any
 *  browser, which shows the tooltips on hover. The structure page is the exception: it draws the
 *  3D structure with 3Dmol.js (BSD-3-Clause), from a CDN or inline.
 */
#pragma once

#include <cstdint>
#include <string>

#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/protein_documents.hpp"
#include "pulsatrix/viz/svg.hpp"

namespace pulsatrix {

/** @brief The 3Dmol.js version the structure page loads. */
inline constexpr const char* k3DmolVersion = "2.5.5";

/**
 * @brief A mutation map: a row per letter of the alphabet, a column per residue, each cell colored
 *        by its score (diverging, scaled to the largest |score|). The wild type's cell has a dot,
 *        and the sequence runs along the top. Unscored cells are gray.
 * @note Up to 12,000 cells are drawn as one rectangle each, with a tooltip such as `K11R: -2.31`;
 *       larger maps are drawn as images, without tooltips.
 * @throws std::invalid_argument for a document ToJson rejects, or unusable options.
 */
[[nodiscard]] std::string RenderMutationMapSvg(const MutationMapDocument& doc, const SvgOptions& options = {});

/**
 * @brief A sequence logo: at each position, the letters stacked to the position's information
 *        content in bits (InformationContent), each as tall as its share, the likeliest on top.
 *        Letters are colored by chemistry, as WebLogo does: hydrophobic black, polar green,
 *        amide purple, basic blue, acidic red. The sequence's own letters run underneath.
 * @note Letters are bold sans-serif text stretched to their box, so a glyph's ink fills it to
 *       within a few percent depending on the installed font (Helvetica, Arial or Liberation Sans).
 * @throws std::invalid_argument for a document ToJson rejects, or unusable options.
 */
[[nodiscard]] std::string RenderSequenceLogoSvg(const SequenceLogoDocument& doc, const SvgOptions& options = {});

/** @brief Options for RenderContactMapSvg. */
struct ContactMapViewOptions {
    /** @brief The best-scored pairs at least this many residues apart are marked: L of them. */
    int64_t min_separation = 6;
};

/**
 * @brief A contact map: one square, residue against residue. Above the diagonal, the predicted
 *        scores (white to dark blue). Below it, the true contacts in gray, and the L best-scored
 *        pairs at least `min_separation` apart as dots: blue where the structure has a contact,
 *        red where it doesn't, hollow where it's unknown. A caption gives the precision at L for
 *        those pairs and for long-range ones (24 apart and more), as ContactPrecision computes it.
 *        Without a structure, the lower triangle shows the dots alone.
 * @throws std::invalid_argument for a document ToJson rejects, min_separation < 1, or unusable
 *         options.
 */
[[nodiscard]] std::string RenderContactMapSvg(const ContactMapDocument& doc, const ContactMapViewOptions& view = {},
                                              const SvgOptions& options = {});

/**
 * @brief Residue tracks: a ruler and the sequence, then a row of colored cells per track (signed
 *        tracks diverging around zero, others Viridis from their minimum or zero), then a row of
 *        labeled spans per feature category. Each track has its own scale, shown in a legend.
 * @throws std::invalid_argument for a document ToJson rejects, or unusable options.
 */
[[nodiscard]] std::string RenderResidueTracksSvg(const ResidueTracksDocument& doc, const SvgOptions& options = {});

// ---- pages ----------------------------------------------------------------------------------
// Each page holds the SVG figure inline; HtmlOptions::width sets its width and title its heading.
// Scripts options are ignored: the pages have none.

[[nodiscard]] std::string RenderMutationMapHtml(const MutationMapDocument& doc, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderSequenceLogoHtml(const SequenceLogoDocument& doc, const HtmlOptions& options = {});
[[nodiscard]] std::string RenderContactMapHtml(const ContactMapDocument& doc, const ContactMapViewOptions& view = {},
                                               const HtmlOptions& options = {});
[[nodiscard]] std::string RenderResidueTracksHtml(const ResidueTracksDocument& doc, const HtmlOptions& options = {});

/**
 * @brief A 3D structure, drawn as a cartoon with 3Dmol.js, with the residues of @p chain colored by
 *        one of @p tracks' tracks at a time (a menu switches between them). Hovering a residue
 *        shows its name, number and value. Other chains are light gray, ligands are sticks, and
 *        water is hidden.
 *
 * Track values reach residues through ResidueTracksDocument::residue_ids when it has them; without
 * them, the chain's residues (ProteinStructure's reading of the file) must have the tracks'
 * sequence, one for one.
 *
 * @param structure The text of a PDB or mmCIF file (mmCIF when it starts with `data_`), embedded
 *                  in the page.
 * @param options HtmlScripts::Cdn loads 3Dmol.js from jsDelivr, pinned and checked with a
 *                Subresource Integrity hash; HtmlScripts::Inline copies `3Dmol-min.js` from
 *                `script_dir` (`tools/render/fetch_vega.sh` downloads it).
 * @note Insertion codes are matched in PDB files only: 3Dmol.js doesn't read them from mmCIF.
 * @throws std::invalid_argument if the structure can't be read, has no chain @p chain, a residue
 *         id isn't in the chain, the sequences don't match, or @p tracks has no track;
 *         std::runtime_error if inline scripts can't be read.
 */
[[nodiscard]] std::string RenderStructureHtml(const std::string& structure, const std::string& chain,
                                              const ResidueTracksDocument& tracks, const HtmlOptions& options = {});

}  // namespace pulsatrix
