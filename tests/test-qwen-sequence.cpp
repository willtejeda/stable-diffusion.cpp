#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#include "model/diffusion/qwen_image_2_1.hpp"

// Uneven image shares and a causal text prefix catch token/RoPE reordering,
// omitted remote keys, and accidental image-to-text attention in Qwen 2.1.
static std::vector<float> run(bool split, bool version21, bool gpu) {
    auto single = gpu ? ggml_backend_init_by_name("CUDA0", nullptr) : sd_backend_cpu_init();
    GGML_ASSERT(single != nullptr);
    if (!gpu) sd_backend_cpu_set_n_threads(single, 1);
    SDTensorSplitContext split_context;
    split_context.sequence = true;
    auto backend = split && gpu ? sd_tensor_split_backend_init(
        {ggml_backend_get_device(single), ggml_backend_dev_by_name("CUDA1")}, &split_context) : single;
    auto weights_ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
    constexpr int64_t dim = 256, head = 128, text_tokens = 3, image_tokens = 7;
    Qwen::QwenImage21Config config;
    config.hidden_size = dim;
    config.head_dim = head;
    config.intermediate_size = 48;
    Qwen::QwenImage21TransformerBlock block(config);
    Qwen::QwenImageAttention attention(dim, head, dim / head);
    GGMLBlock& module = version21 ? static_cast<GGMLBlock&>(block) : static_cast<GGMLBlock&>(attention);
    module.init(weights_ctx);
    std::map<std::string, ggml_tensor*> weights;
    module.get_param_tensors(weights);
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 0;
    for (const auto& item : weights) {
        std::vector<float> values(ggml_nelements(item.second));
        for (auto& value : values) {
            value = item.first.find("norm") != std::string::npos ? 1.f : 0.1f * std::sin(float(++seed));
        }
        ggml_backend_tensor_set(item.second, values.data(), 0, values.size() * sizeof(float));
    }
    auto img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, image_tokens);
    auto txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, text_tokens);
    auto mods = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim * 4, 2);
    auto mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, text_tokens, text_tokens);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, head / 2, text_tokens + image_tokens);
    auto inputs_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto tensor : {img, txt, mods}) {
        std::vector<float> values(ggml_nelements(tensor));
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            values[i] = std::cos(float(i) * 0.17f) * (tensor == mods ? 0.3f : 1.f);
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    std::vector<float> mask_values(text_tokens * text_tokens);
    for (int q = 0; q < text_tokens; ++q) {
        for (int k = q + 1; k < text_tokens; ++k) mask_values[k + q * text_tokens] = -INFINITY;
    }
    ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size() * sizeof(float));
    std::vector<std::vector<float>> positions;
    for (int i = 0; i < text_tokens + image_tokens; ++i) positions.push_back({float(i), float(i % 3), float(i % 5)});
    auto pe_values = Rope::embed_nd(positions, 1, 10000.f, {32, 48, 48});
    ggml_backend_tensor_set(pe, pe_values.data(), 0, pe_values.size() * sizeof(float));
    GGMLRunnerContext runner;
    runner.backend = backend;
    runner.ggml_ctx = ctx;
    runner.flash_attn_enabled = true;
    auto graph = ggml_new_graph_custom(ctx, 4096, false);
    runner.graph = graph;
    ggml_tensor* out_img;
    ggml_tensor* out_txt;
    if (split) {
        auto txt_pe = ggml_ext_slice(ctx, pe, 3, 0, text_tokens);
        auto img_pe = ggml_ext_slice(ctx, pe, 3, text_tokens, text_tokens + image_tokens);
        img = sd_seq_split(ctx, backend, img, image_tokens);
        auto local_pe = sd_seq_split(ctx, backend, img_pe, image_tokens);
        if (version21) {
            auto modulation = ggml_ext_chunk(ctx, mods, 4, 0);
            auto result = block.forward_sequence(&runner, img, txt, modulation, local_pe, img_pe, txt_pe, mask);
            out_img = result.first;
            out_txt = result.second;
        } else {
            auto result = attention.forward(&runner, img, txt, pe, nullptr, local_pe);
            out_img = result.first;
            out_txt = result.second;
        }
        out_img = sd_seq_gather(ctx, backend, out_img);
    } else if (version21) {
        Qwen::QwenImage21Layout layout;
        layout.prefix_length = text_tokens;
        layout.segments = {{0, text_tokens, 0, -1}, {text_tokens, text_tokens + image_tokens, text_tokens, 0}};
        auto joint = ggml_concat(ctx, txt, img, 1);
        joint = block.forward(&runner, joint, ggml_ext_chunk(ctx, mods, 4, 0), pe, layout, {mask, nullptr}, {});
        out_txt = ggml_ext_slice(ctx, joint, 1, 0, text_tokens);
        out_img = ggml_ext_slice(ctx, joint, 1, text_tokens, text_tokens + image_tokens);
    } else {
        auto result = attention.forward(&runner, img, txt, pe);
        out_img = result.first;
        out_txt = result.second;
    }
    auto out = ggml_concat(ctx, out_txt, out_img, 1);
    ggml_build_forward_expand(graph, out);
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(allocator, graph));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(inputs_buffer);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(ctx);
    ggml_free(weights_ctx);
    // Token index tables are cached per backend for process lifetime.
    if (!split || !gpu) ggml_backend_free(single);
    return result;
}

int main(int argc, char** argv) {
    const bool gpu = argc == 2 && std::string(argv[1]) == "--cuda";
    for (bool version21 : {false, true}) {
        auto reference = run(false, version21, gpu);
        auto split = run(true, version21, gpu);
        float max_error = 0.f;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(split[i])) return 1;
            max_error = std::max(max_error, std::abs(reference[i] - split[i]));
        }
        std::printf("Qwen %s whole vs sequence: max absolute error %.9g\n", version21 ? "2.1 causal block" : "dual-stream attention", max_error);
        if (max_error > 0.002f) return 2;
    }
    return 0;
}
