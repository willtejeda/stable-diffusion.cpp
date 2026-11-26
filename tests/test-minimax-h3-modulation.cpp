#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-alloc.h"
#include "model/diffusion/minimax_h3.hpp"
#include "runtime/denoiser.hpp"

// Exercise token-indexed adaLN across text-tag, reference, audio and video boundaries,
// including a segment crossing the midpoint and an odd token count.
static int check_block(bool curves) {
    auto backend = sd_backend_cpu_init();
    sd_backend_cpu_set_n_threads(backend, 1);
    auto ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
    MiniMaxH3::Config config;
    config.hidden_size = 32;
    config.num_attention_heads = 2;
    config.attention_head_dim = 16;
    config.ffn_hidden_size = 48;
    config.adaln_curve_grid = curves ? 4 : 0;
    config.time_embed_dim = 16;
    MiniMaxH3::TransformerBlock block(config);
    block.init(ctx);
    std::map<std::string, ggml_tensor*> weights;
    block.get_param_tensors(weights);
    int seed = 0;
    for (const auto& item : weights) {
        auto tensor = item.second;
        auto data = static_cast<float*>(tensor->data);
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            data[i] = item.first.find("norm.weight") != std::string::npos ||
                              item.first.find("norm1.weight") != std::string::npos ||
                              item.first.find("norm2.weight") != std::string::npos
                          ? 1.f : 0.04f * std::sin(float(++seed));
        }
    }
    constexpr int64_t tokens = 17;
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, tokens);
    auto time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.time_embed_dim, 4);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        static_cast<float*>(x->data)[i] = std::cos(float(i) * 0.13f);
    }
    for (int64_t i = 0; i < ggml_nelements(time); ++i) {
        static_cast<float*>(time->data)[i] = std::sin(float(i) * 0.31f);
    }
    const std::vector<MiniMaxH3::TokenModulationSpan> spans = {
        {0, 2, 1}, {2, 4, 2}, {4, 6, 6}, {6, 11, 11}, {11, tokens, 0}};
    auto index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    const auto indices = MiniMaxH3::modulation_indices(spans, tokens);
    std::memcpy(index->data, indices.data(), indices.size() * sizeof(int32_t));
    GGMLRunnerContext runner;
    runner.backend = backend;
    runner.ggml_ctx = ctx;
    auto reference = block.forward(&runner, x, time, spans, nullptr);
    auto indexed = block.forward(&runner, x, time, spans, nullptr, index);
    auto graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, indexed);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        return 1;
    }
    float max_error = 0.f;
    for (int64_t i = 0; i < ggml_nelements(reference); ++i) {
        float a = static_cast<float*>(reference->data)[i];
        float b = static_cast<float*>(indexed->data)[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            return 2;
        }
        max_error = std::max(max_error, std::abs(a - b));
    }
    std::printf("%s segment vs token-indexed block: max absolute error %.9g\n",
                curves ? "curve" : "linear", max_error);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return max_error <= 1e-5f ? 0 : 3;
}

// The six-step checkpoint stores one base prediction and five residual heads.
// Test the actual final projection with an odd token count and a reference block.
static int check_image() {
    MiniMaxH3ImageDenoiser denoiser;
    const auto sigmas = denoiser.get_sigmas(6, 1024, DISCRETE_SCHEDULER, VERSION_MINIMAX_H3_IMAGE);
    const std::vector<float> expected = {1.f, 0.967523694f, 0.887548506f, 0.75f, 0.532773912f, 0.232008874f, 0.f};
    if (sigmas != expected) {
        return 4;
    }
    const auto exponential = denoiser.get_sigmas(6, 1024, EXPONENTIAL_SCHEDULER, VERSION_MINIMAX_H3_IMAGE);
    for (size_t i = 0; i + 1 < exponential.size(); ++i) {
        if (!std::isfinite(exponential[i]) || exponential[i] <= exponential[i + 1]) {
            std::printf("image exponential scheduler: nonfinite or nondecreasing sigma at %zu\n", i);
            return 9;
        }
    }
    denoiser.set_shift(12.f);
    const auto pdmd = denoiser.get_sigmas(2, 1024, SIMPLE_SCHEDULER, VERSION_MINIMAX_H3_IMAGE);
    if (pdmd.size() != 3 || pdmd.front() != 1.f || std::abs(pdmd[1] - 12.f / 13.f) > 1e-6f || pdmd.back() != 0.f) {
        std::printf("image pDMD simple scheduler: unexpected two-step schedule\n");
        return 10;
    }

    const std::vector<sd::Tensor<float>> refs = {sd::zeros<float>({4, 2, 1, 24, 1})};
    const std::vector<MiniMaxH3ReferenceBlock> ref_blocks = {{MiniMaxH3ReferenceKind::IMAGE, 0, -1}};
    const auto layout = MiniMaxH3::build_layout(3, 1, 2, 6, 0, refs, {}, {}, ref_blocks, {}, 0.f, 0.f);
    const auto indices = MiniMaxH3::modulation_indices(layout.segments, 8);
    if (layout.sequence_segments.size() != 3 || layout.video_segment.start != 5 ||
        layout.video_segment.end != 8 || indices != std::vector<int32_t>{1, 1, 1, 3, 3, 0, 0, 0} ||
        layout.positions[3 * 5] != 4.f) {
        return 5;
    }

    auto backend = sd_backend_cpu_init();
    sd_backend_cpu_set_n_threads(backend, 1);
    auto ctx = ggml_init({8 * 1024 * 1024, nullptr, false});
    MiniMaxH3::Config config;
    config.hidden_size = 32;
    config.time_embed_dim = 8;
    config.adaln_curve_grid = 4;
    config.audio_latent_channels = 0;
    config.pdd_head_count = 6;
    MiniMaxH3::FinalLayer final(config);
    final.init(ctx);
    std::map<std::string, ggml_tensor*> params;
    final.get_param_tensors(params);
    for (const auto& [name, tensor] : params) {
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
        if (name == "video_out.bias") {
            auto* bias = static_cast<float*>(tensor->data);
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                bias[i] = 1.f + i / 96;
            }
        }
    }
    auto* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, 8);
    auto* time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.time_embed_dim, 3);
    std::memset(x->data, 0, ggml_nbytes(x));
    std::memset(time->data, 0, ggml_nbytes(time));
    GGMLRunnerContext runner;
    runner.backend = backend;
    runner.ggml_ctx = ctx;
    const auto first = final.forward(&runner, x, time, layout.video_segment, {}, sigmas[0], sigmas[1]);
    const auto last = final.forward(&runner, x, time, layout.video_segment, {}, sigmas[5], sigmas[6]);
    const auto coarse = final.forward(&runner, x, time, layout.video_segment, {}, sigmas[3], sigmas[6]);
    auto* graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, first.first);
    ggml_build_forward_expand(graph, last.first);
    ggml_build_forward_expand(graph, coarse.first);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    const float coarse_value = 1.f + (4.f * (0.75f - 0.532773912f) +
                                      5.f * (0.532773912f - 0.232008874f) + 6.f * 0.232008874f) / 0.75f;
    bool correct = computed;
    for (const auto& [result, value] : {std::pair{first, 1.f}, std::pair{last, 7.f}, std::pair{coarse, coarse_value}}) {
        auto* tensor = result.first;
        correct = correct && !result.second && tensor->ne[0] == 96 && tensor->ne[1] == 3 && ggml_is_contiguous(tensor);
        if (correct) {
            const auto* data = static_cast<const float*>(tensor->data);
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                correct = correct && std::abs(data[i] - value) <= 1e-5f;
            }
        }
    }
    std::printf("image PDD heads and reference layout: %s\n", correct ? "OK" : "FAIL");
    ggml_free(ctx);
    ggml_backend_free(backend);
    return correct ? 0 : 6;
}

// CPU meta devices have no backend events. Unequal slices expose wrong offset
// and duplicate-slice bugs in the host-staged sequence gather.
static int check_host_gather() {
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = {cpu, cpu};
    auto set_layout = [](const ggml_tensor* tensor, void*) {
        ggml_backend_meta_split_state state{};
        state.n_segments = 1;
        state.nr[0] = 1;
        if (std::strcmp(tensor->name, "input") == 0) {
            state.axis = GGML_BACKEND_SPLIT_AXIS_1;
            state.ne[0] = 2;
            state.ne[1] = 3;
        } else {
            state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        }
        return state;
    };
    auto device = ggml_backend_meta_device(devices, 2, set_layout, nullptr);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    auto weights = ggml_init({1024 * 1024, nullptr, true});
    auto* input = ggml_new_tensor_2d(weights, GGML_TYPE_F32, 2, 5);
    ggml_set_name(input, "input");
    auto* output = ggml_backend_meta_all_gather(ctx, ggml_scale(ctx, input, 2.f));
    auto* graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        return 8;
    }
    const float samples[10] = {-3, 7, 5, 1, 8, -2, 3, 4, 9, -6};
    ggml_backend_tensor_set(input, samples, 0, sizeof(samples));
    float values[10]{};
    bool correct = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (correct) {
        ggml_backend_tensor_get(output, values, 0, sizeof(values));
        for (size_t i = 0; i < 10; ++i) {
            correct = correct && values[i] == 2.f * samples[i];
        }
    }
    std::printf("host sequence gather, unequal CPU shards: %s\n", correct ? "OK" : "FAIL");
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_free(weights);
    ggml_backend_free(backend);
    return correct ? 0 : 7;
}

int main() {
    const int linear = check_block(false);
    if (linear != 0) {
        return linear;
    }
    const int curves = check_block(true);
    if (curves != 0) {
        return curves;
    }
    const int image = check_image();
    return image != 0 ? image : check_host_gather();
}