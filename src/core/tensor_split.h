#ifndef __SD_CORE_TENSOR_SPLIT_H__
#define __SD_CORE_TENSOR_SPLIT_H__

// Tensor parallelism (--split-mode <module>=tensor): one ggml meta device over the module's GPUs. Weights in its
// buffer type are sliced across the cards by sd_tensor_split_state(); graphs computed on its backend are split op by
// op and all-reduced where partial sums meet (ggml/src/ggml-backend-meta.cpp). Everything not named by a rule is
// mirrored (a full copy on every card).

#include <cstddef>
#include <vector>

#include "ggml-backend.h"

// User data for the meta device's split callback: how many cards share a split tensor.
struct SDTensorSplitContext {
    size_t n_devices = 0;
};

// Stage B, the video attention split by heads: opt-in with SD_TENSOR_SPLIT_ATTN=1 while its head selection (full Q/K
// transposes) costs more than it saves (Video/tensor-parallel-plan.md).
bool sd_tensor_split_attention_enabled();

// The meta device's split callback (ggml_backend_meta_get_split_state_t).
ggml_backend_meta_split_state sd_tensor_split_state(const ggml_tensor* tensor, void* userdata);

// A meta backend's per-card backends (owned by it).
std::vector<ggml_backend_t> sd_tensor_split_card_backends(ggml_backend_t meta);

// A meta backend over devices, or nullptr. ctx must outlive the backend.
ggml_backend_t sd_tensor_split_backend_init(const std::vector<ggml_backend_dev_t>& devices, SDTensorSplitContext* ctx);

// True for a backend made by sd_tensor_split_backend_init(): model code must not emit sd.cpp's fused extension ops
// for it (the meta backend cannot split an op it does not know).
bool sd_backend_is_tensor_split(ggml_backend_t backend);

// True for a buffer type of a meta device (a tensor-split backend's weights or compute buffers).
bool sd_buft_is_tensor_split(ggml_backend_buffer_type_t buft);

// A static I32 table 0..n-1 on a tensor-split backend, split across its cards like a column-split projection's
// outputs; cached per backend and n. nullptr for any other backend.
ggml_tensor* sd_tensor_split_feature_index(ggml_backend_t backend, int64_t n);

// On a tensor-split backend: x ([features, tokens, batch], mirrored) reduced to each card's share of the features,
// laid out as a column-split projection's output. Elsewhere: x unchanged.
ggml_tensor* sd_tensor_split_select_features(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* x);

#endif
