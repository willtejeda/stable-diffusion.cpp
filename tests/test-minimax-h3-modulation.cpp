#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model/diffusion/minimax_h3.hpp"

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

int main() {
    const int linear = check_block(false);
    return linear != 0 ? linear : check_block(true);
}