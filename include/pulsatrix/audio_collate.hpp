/** @file audio_collate.hpp
 *  @brief AudioPadCollate -- zero-pads variable-length waveforms into one batch Tensor.
 *  @ingroup data_pipeline
 */
#pragma once

#include "pulsatrix/collate.hpp"

namespace pulsatrix {

/**
 * @brief Builds a CollateFn that zero-pads (silence) variable-length waveforms
 *        (sample.fields[0], shape (1, channels, num_samples) -- WavReader's/
 *        AudioFolderDataset's convention) to the batch's own max sample count, producing
 *        one (N, channels, max_samples) Tensor, plus a (N,) length field. Phase 4's version
 *        of PadCollate (Phase 3, text) -- a second proof that the CollateFn extension point
 *        handles ragged/variable-length modalities with zero Dataset/DataLoader/Batch
 *        interface changes.
 * @return A CollateFn producing Batch{ {padded_waveforms (N,channels,max_samples), lengths (N,)} }.
 * @throws std::invalid_argument (from the returned CollateFn, at call time) if samples is
 *         empty, or if samples have differing channel counts -- external boundary,
 *         matching DefaultCollate's/PadCollate's own convention.
 */
[[nodiscard]] CollateFn AudioPadCollate();

}  // namespace pulsatrix
