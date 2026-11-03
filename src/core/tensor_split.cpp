#include "core/tensor_split.h"

#include <cstdlib>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_set>

#include "core/util.h"

namespace {

    // Which weights are split, and along which ggml axis. A weight is [ne0 = in, ne1 = out]:
    //   axis 1 (columns): each card computes part of the outputs (Q/K/V, FFN up); the result is split along ne0.
    //   axis 0 (rows):    each card takes its part of the inputs (out-projection, FFN down); the result is PARTIAL
    //                     and the meta backend all-reduces it.
    struct SplitRule {
        std::regex pattern;
        int axis;
    };

    const std::vector<SplitRule>& split_rules() {
        // Stage A (Video/tensor-parallel-plan.md): the video blocks' feed-forward only. LTX-2.x: GELU FFN without
        // biases, ff.net.0.proj [dim -> 4 dim], ff.net.2 [4 dim -> dim]; the audio branch (audio_ff) stays mirrored.
        static const std::vector<SplitRule> rules = {
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.ff\.net\.0\.proj\.weight$)"), 1},
            {std::regex(R"((^|\.)transformer_blocks\.\d+\.ff\.net\.2\.weight$)"), 0},
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
