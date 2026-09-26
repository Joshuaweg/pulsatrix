/** @file text_collate.hpp
 *  @brief PadCollate -- right-pads variable-length token sequences into one batch Tensor.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/collate.hpp"

namespace pulsatrix {

/**
 * @brief Builds a CollateFn that right-pads variable-length token sequences
 *        (sample.fields[0], shape (1, seq_len) -- TextDataset::get()'s output) to the
 *        batch's own max length, producing one (N, max_len) Tensor, plus a (N,) length
 *        field recording each sample's real (pre-padding) length. This is the CollateFn
 *        extension point Phase 1's architecture design reserved for ragged/variable-length
 *        modalities (PyTorch's collate_fn equivalent) -- exercised here for the first time,
 *        with zero changes needed to Dataset/DataLoader/Batch themselves.
 * @param pad_index Value used to fill padding positions (as a float, matching Decision
 *        Point 6's token-as-float32 representation). Defaults to 0 -- note this numerically
 *        collides with <unk>'s reserved index; a consumer that needs to distinguish real
 *        <unk> tokens from padding must use the length field, not the token value itself,
 *        to locate padding positions.
 * @return A CollateFn producing Batch{ {padded_tokens (N,max_len), lengths (N,)} }.
 * @throws std::invalid_argument (from the returned CollateFn, at call time) if samples is
 *         empty -- matching DefaultCollate's own external-boundary convention.
 */
[[nodiscard]] CollateFn PadCollate(float pad_index = 0.0f);

}  // namespace pulsatrix
