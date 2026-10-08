// What's wrong with a protein viz document, as "<JSON Pointer>: <problem>", or empty when nothing
// is. Shared by the documents' readers and writers and by the renderers. Private to src/.
#pragma once

#include <string>

#include "pulsatrix/viz/protein_documents.hpp"

namespace pulsatrix {
namespace protein_detail {

std::string MutationMapProblem(const MutationMapDocument& doc);
std::string SequenceLogoProblem(const SequenceLogoDocument& doc);
std::string ContactMapProblem(const ContactMapDocument& doc);
std::string ResidueTracksProblem(const ResidueTracksDocument& doc);

}  // namespace protein_detail
}  // namespace pulsatrix
