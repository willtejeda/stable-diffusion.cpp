#include "core/tensor_split.h"

#include <cstdlib>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "core/util.h"

// From ggml-backend-impl.h (internal, not on sd.cpp's include path): a meta backend's simple backends.
extern "C" {
GGML_API size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend);
GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);
}

namespace {

    // Which weights are split, and along which ggml axis. A weight is [ne0 = in, ne1 = out]:
    //   axis 1 (columns): each card computes part of the outputs (Q/K/V, FFN up); the result is split along ne0.
    //   axis 0 (rows):    each card takes its part of the inputs (out-projection, FFN down); the result is PARTIAL
    //                     and the meta backend all-reduces it.
    struct SplitRule {
        std::regex pattern;
        int axis;
        bool attention = false;  // only with SD_TENSOR_SPLIT_ATTN=1 (stage B, slower than stage A for now)
    };

    const std::vector<SplitRule>& split_rules() {
        // Stage A (Video/tensor-parallel-plan.md): the video blocks' feed-forward only. LTX-2.x: GELU FFN without
        // biases, ff.net.0.proj [dim -> 4 dim], ff.net.2 [4 dim -> dim]; the audio branch (audio_ff) stays mirrored.
        // Stage B: the video blocks' attention by heads. Q and K are projected, normed and rotated whole on every
        // card (full-width Q/K RMSNorm and per-head RoPE stay fused), then each card keeps its heads
        // (sd_tensor_split_select_features); V, the per-head gates and the out-projection are split.
        static const std::vector<SplitRule> rules = {
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.ff\.net\.0\.proj\.weight$)"), 1},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.ff\.net\.2\.weight$)"), 0},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.attn[12]\.to_v\.weight$)"), 1, true},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.attn[12]\.to_v\.bias$)"), 0, true},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.attn[12]\.to_gate_logits\.weight$)"), 1, true},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.attn[12]\.to_gate_logits\.bias$)"), 0, true},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.attn[12]\.to_out\.0\.weight$)"), 0, true},
            // the feature index tables of sd_tensor_split_select_features: split like the outputs above
            {std::regex(R"(^sd_tensor_split\.features\.\d+$)"), 0},
        };
        return rules;
    }

    std::mutex g_mutex;
    std::unordered_set<ggml_backend_t> g_tensor_split_backends;

    ggml_backend_meta_split_state mirrored() {
        ggml_backend_meta_split_state state = {};
        state.axis                          = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        state.nr[0]                         = 1;
        state.n_segments                    = 1;
        return state;
    }

    // Equal parts along axis, each a whole number of quantisation blocks along ne0 (a block never straddles cards).
    bool split_evenly(const ggml_tensor* tensor, int axis, size_t n_devices, ggml_backend_meta_split_state* out) {
        const int64_t total       = tensor->ne[axis];
        const int64_t granularity = axis == 0 ? ggml_blck_size(tensor->type) : 1;
        if (n_devices < 2 || total % granularity != 0) {
            return false;
        }
        const int64_t units = total / granularity;
        if (units < (int64_t)n_devices) {
            return false;
        }
        ggml_backend_meta_split_state state = {};
        state.axis                          = (ggml_backend_meta_split_axis)axis;
        state.nr[0]                         = 1;
        state.n_segments                    = 1;
        int64_t given                       = 0;
        for (size_t j = 0; j < n_devices; j++) {
            const int64_t share = j + 1 == n_devices ? units - given : units / (int64_t)n_devices;
            state.ne[j]         = share * granularity;
            given += share;
        }
        *out = state;
        return true;
    }

}  // namespace

bool sd_tensor_split_attention_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("SD_TENSOR_SPLIT_ATTN");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
}

ggml_backend_meta_split_state sd_tensor_split_state(const ggml_tensor* tensor, void* userdata) {
    const auto* ctx = static_cast<const SDTensorSplitContext*>(userdata);
    if (ctx == nullptr || tensor == nullptr) {
        return mirrored();
    }
    static const bool mirror_all = [] {
        const char* v = std::getenv("SD_TENSOR_SPLIT_MIRROR_ALL");  // diagnostics: the meta device, nothing split
        return v != nullptr && v[0] == '1';
    }();
    if (mirror_all) {
        return mirrored();
    }
    const std::string name = tensor->name;
    for (const SplitRule& rule : split_rules()) {
        if (rule.attention && !sd_tensor_split_attention_enabled()) {
            continue;
        }
        if (!std::regex_search(name, rule.pattern)) {
            continue;
        }
        ggml_backend_meta_split_state state;
        if (split_evenly(tensor, rule.axis, ctx->n_devices, &state)) {
            return state;
        }
        LOG_WARN("tensor split: %s [%lld, %lld] (%s) cannot be split evenly along axis %d; mirrored",
                 name.c_str(),
                 (long long)tensor->ne[0],
                 (long long)tensor->ne[1],
                 ggml_type_name(tensor->type),
                 rule.axis);
        break;
    }
    return mirrored();
}

ggml_backend_t sd_tensor_split_backend_init(const std::vector<ggml_backend_dev_t>& devices, SDTensorSplitContext* ctx) {
    if (devices.size() < 2 || ctx == nullptr) {
        return nullptr;
    }
    ctx->n_devices = devices.size();
    std::vector<ggml_backend_dev_t> devs(devices);
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs.data(), devs.size(), sd_tensor_split_state, ctx);
    if (meta == nullptr) {
        return nullptr;
    }
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);
    if (backend != nullptr) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_tensor_split_backends.insert(backend);
    }
    return backend;
}

bool sd_backend_is_tensor_split(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    // Diagnostics: SD_NO_EXT_OPS=1 takes the plain-ggml paths on any backend, as a tensor-split backend does, so a
    // one-card run can be compared with a tensor-split run op for op.
    static const bool no_ext_ops = [] {
        const char* v = std::getenv("SD_NO_EXT_OPS");
        return v != nullptr && v[0] == '1';
    }();
    if (no_ext_ops) {
        return true;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_tensor_split_backends.count(backend) != 0;
}

bool sd_buft_is_tensor_split(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && std::string(ggml_backend_reg_name(reg)) == "Meta";
}

namespace {
    struct FeatureIndexCache {
        std::mutex mutex;
        std::unordered_map<ggml_backend_t, std::unordered_map<int64_t, ggml_tensor*>> tables;
        std::vector<ggml_context*> contexts;  // kept for the life of the process, like the tables
    };

    FeatureIndexCache& feature_index_cache() {
        static FeatureIndexCache cache;
        return cache;
    }
}  // namespace

ggml_tensor* sd_tensor_split_feature_index(ggml_backend_t backend, int64_t n) {
    if (!sd_backend_is_tensor_split(backend) || n <= 0) {
        return nullptr;
    }
    auto& cache = feature_index_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto& tables = cache.tables[backend];
    auto it      = tables.find(n);
    if (it != tables.end()) {
        return it->second;
    }
    ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
    ggml_context* ctx       = ggml_init(params);
    if (ctx == nullptr) {
        return nullptr;
    }
    ggml_tensor* table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(table, ("sd_tensor_split.features." + std::to_string(n)).c_str());
    // A weights-like buffer: the split callback places each card's half of the indices on that card.
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    if (buffer == nullptr) {
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<int32_t> values(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(table, values.data(), 0, values.size() * sizeof(int32_t));
    cache.contexts.push_back(ctx);
    tables[n] = table;
    return table;
}

ggml_tensor* sd_tensor_split_select_features(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* x) {
    ggml_tensor* index = sd_tensor_split_feature_index(backend, x->ne[0]);
    if (index == nullptr) {
        return x;
    }
    // x: [features, tokens, batch], whole on every card. Transposed, a feature is a row: each card gathers the rows
    // its index half names, and the transpose back is split along the features like a column-split projection.
    const int64_t features = x->ne[0];
    const int64_t rows     = x->ne[1] * x->ne[2] * x->ne[3];
    ggml_tensor* t         = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, x), features, rows)));
    ggml_tensor* picked    = ggml_get_rows(ctx, t, index);  // [rows, features], split along the features
    ggml_tensor* back      = ggml_cont(ctx, ggml_transpose(ctx, picked));
    return ggml_reshape_4d(ctx, back, features, x->ne[1], x->ne[2], x->ne[3]);
}

std::vector<ggml_backend_t> sd_tensor_split_card_backends(ggml_backend_t meta) {
    std::vector<ggml_backend_t> cards;
    if (!sd_backend_is_tensor_split(meta)) {
        return cards;
    }
    for (size_t i = 0; i < ggml_backend_meta_n_backends(meta); i++) {
        cards.push_back(ggml_backend_meta_simple_backend(meta, i));
    }
    return cards;
}
