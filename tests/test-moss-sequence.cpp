#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "model/diffusion/moss_image.hpp"

// Exercise both the public model output and the image/text streams across two blocks.
static std::vector<float> run(bool split, bool gpu, bool model_output) {
    auto single = gpu ? ggml_backend_init_by_name("CUDA0", nullptr) : sd_backend_cpu_init();
    GGML_ASSERT(single != nullptr);
    if (!gpu) sd_backend_cpu_set_n_threads(single, 1);
    SDTensorSplitContext split_context;
    split_context.sequence = true;
    auto backend = split && gpu ? sd_tensor_split_backend_init(
        {ggml_backend_get_device(single), ggml_backend_dev_by_name("CUDA1")}, &split_context) : single;
    GGML_ASSERT(backend != nullptr);
    for (auto card : sd_tensor_split_card_backends(backend)) sd_ggml_register_extensions(card);
    auto weights_ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    auto ctx = ggml_init({64 * 1024 * 1024, nullptr, true});
    constexpr int64_t dim = 256, head = 128, image_tokens = 7, text_tokens = 3;
    MOSSImage::MOSSImageConfig config;
    config.dim = dim;
    config.head_dim = head;
    config.ffn_dim = 48;
    config.channels = 4;
    config.text_dim = 32;
    config.freq_dim = 16;
    config.depth = 2;
    MOSSImage::Model model(config);
    MOSSImage::Block first(config), second(config);
    std::map<std::string, ggml_tensor*> weights;
    if (model_output) {
        model.init(weights_ctx);
        model.get_param_tensors(weights);
    } else {
        first.init(weights_ctx);
        second.init(weights_ctx);
        first.get_param_tensors(weights, "first.");
        second.get_param_tensors(weights, "second.");
    }
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 0;
    for (const auto& item : weights) {
        std::vector<float> values(ggml_nelements(item.second));
        for (auto& value : values) value = item.first.find("norm") != std::string::npos ? 1.f : 0.03f * std::sin(float(++seed));
        ggml_backend_tensor_set(item.second, values.data(), 0, values.size() * sizeof(float));
    }
    auto img = model_output ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, image_tokens, 1, config.channels, 1)
                            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, image_tokens);
    auto txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model_output ? config.text_dim : dim, text_tokens);
    auto time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.freq_dim, 1);
    std::array<ggml_tensor*, 6> mod;
    for (auto& part : mod) part = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, head / 2, image_tokens + text_tokens);
    auto inputs_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    int input_seed = 0;
    auto initialize = [&](ggml_tensor* tensor) {
        std::vector<float> values(ggml_nelements(tensor));
        for (auto& value : values) value = 0.4f * std::cos(float(++input_seed) * 0.17f);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    };
    initialize(img);
    initialize(txt);
    initialize(time);
    for (auto part : mod) initialize(part);
    std::vector<std::vector<float>> positions;
    for (int i = 0; i < image_tokens + text_tokens; ++i) {
        positions.push_back({float((i * 7) % 11), float(i % 3), float(i * i % 13), float((i * 3) % 5)});
    }
    auto pe_values = Rope::embed_nd(positions, 1, 10000.f, {32, 32, 32, 32});
    ggml_backend_tensor_set(pe, pe_values.data(), 0, pe_values.size() * sizeof(float));
    GGMLRunnerContext runner;
    runner.backend = backend;
    runner.ggml_ctx = ctx;
    runner.flash_attn_enabled = true;
    auto graph = ggml_new_graph_custom(ctx, 16384, false);
    runner.graph = graph;
    ggml_tensor* out;
    if (model_output) {
        out = model.forward(&runner, img, time, txt, pe, {});
    } else if (split) {
        auto image_pe = ggml_ext_slice(ctx, pe, 3, 0, image_tokens);
        auto text_pe = ggml_ext_slice(ctx, pe, 3, image_tokens, image_tokens + text_tokens);
        auto local_pe = sd_seq_split(ctx, backend, image_pe, image_tokens);
        img = sd_seq_split(ctx, backend, img, image_tokens);
        auto streams = first.forward_sequence(&runner, img, txt, local_pe, image_pe, text_pe, mod);
        streams = second.forward_sequence(&runner, streams.first, streams.second, local_pe, image_pe, text_pe, mod);
        out = ggml_concat(ctx, sd_seq_gather(ctx, backend, streams.first), streams.second, 1);
    } else {
        out = ggml_concat(ctx, img, txt, 1);
        out = first.forward(&runner, out, pe, mod);
        out = second.forward(&runner, out, pe, mod);
    }
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
    // Sequence token index tables are cached for the lifetime of the backend.
    if (!split || !gpu) ggml_backend_free(single);
    return result;
}

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--cuda")) {
        std::fprintf(stderr, "usage: %s [--cuda]\n", argv[0]);
        return 1;
    }
    const bool gpu = argc == 2;
    for (bool model_output : {false, true}) {
        auto reference = run(false, gpu, model_output);
        auto split = run(true, gpu, model_output);
        if (reference.size() != split.size()) return 1;
        float max_error = 0.f;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(split[i])) return 1;
            max_error = std::max(max_error, std::abs(reference[i] - split[i]));
        }
        std::printf("MOSS %s whole vs sequence: max absolute error %.9g\n",
                    model_output ? "model output" : "image and text blocks", max_error);
        if (max_error > 0.002f) return 2;
    }
    return 0;
}
