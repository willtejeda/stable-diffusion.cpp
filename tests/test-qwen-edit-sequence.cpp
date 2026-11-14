#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "model/diffusion/qwen_image.hpp"

// The target/reference boundary crosses an uneven token partition. Different
// timestep rows and nonuniform RoPE positions expose lost reference modulation.
static std::vector<float> run(bool split, bool gpu, bool zero_cond_t) {
    auto single = gpu ? ggml_backend_init_by_name("CUDA0", nullptr) : sd_backend_cpu_init();
    GGML_ASSERT(single != nullptr);
    if (!gpu) sd_backend_cpu_set_n_threads(single, 1);
    SDTensorSplitContext split_context;
    split_context.sequence = true;
    auto backend = split && gpu ? sd_tensor_split_backend_init(
        {ggml_backend_get_device(single), ggml_backend_dev_by_name("CUDA1")}, &split_context) : single;
    GGML_ASSERT(backend != nullptr);
    for (auto card : sd_tensor_split_card_backends(backend)) sd_ggml_register_extensions(card);
    auto weights_ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
    Qwen::QwenImageConfig config;
    config.patch_size = 1;
    config.in_channels = 4;
    config.out_channels = 4;
    config.num_layers = 2;
    config.num_attention_heads = 2;
    config.joint_attention_dim = 32;
    config.zero_cond_t = zero_cond_t;
    Qwen::QwenImageModel model(config);
    model.init(weights_ctx);
    std::map<std::string, ggml_tensor*> weights;
    model.get_param_tensors(weights);
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 0;
    for (const auto& item : weights) {
        std::vector<float> values(ggml_nelements(item.second));
        const bool norm_weight = item.first.find("norm") != std::string::npos &&
                                 item.first.find("linear") == std::string::npos;
        for (auto& value : values) value = norm_weight ? 1.f : 0.03f * std::sin(float(++seed) * 0.13f);
        ggml_backend_tensor_set(item.second, values.data(), 0, values.size() * sizeof(float));
    }
    auto target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 4, 1);
    auto reference = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, 4, 1);
    auto text = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 3);
    auto timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto index = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 64, 10);
    auto one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(one, "ggml_runner_build_in_tensor:one");
    auto inputs_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    int input_seed = 0;
    for (auto tensor : {target, reference, text}) {
        std::vector<float> values(ggml_nelements(tensor));
        for (auto& value : values) value = 0.4f * std::cos(float(++input_seed) * 0.17f);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    const float time = 0.73f;
    ggml_backend_tensor_set(timestep, &time, 0, sizeof(time));
    const float indices[] = {0, 0, 0, 0, 1, 1, 1};
    const float unity = 1.f;
    ggml_backend_tensor_set(one, &unity, 0, sizeof(unity));
    ggml_backend_tensor_set(index, indices, 0, sizeof(indices));
    std::vector<std::vector<float>> positions;
    for (int i = 0; i < 10; ++i) positions.push_back({float(i >= 7), float(i % 3), float(i * 2)});
    auto pe_values = Rope::embed_nd(positions, 1, 10000.f, {32, 48, 48});
    ggml_backend_tensor_set(pe, pe_values.data(), 0, pe_values.size() * sizeof(float));
    GGMLRunnerContext runner;
    runner.backend = backend;
    runner.ggml_ctx = ctx;
    runner.flash_attn_enabled = true;
    auto graph = ggml_new_graph_custom(ctx, 8192, false);
    runner.graph = graph;
    auto out = model.forward(&runner, target, timestep, nullptr, text, pe, {reference}, zero_cond_t ? index : nullptr);
    GGML_ASSERT(ggml_nelements(out) == ggml_nelements(target));
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
    // Meta token tables are cached by backend identity for process lifetime.
    if (!split || !gpu) ggml_backend_free(single);
    return result;
}

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--cuda")) return 1;
    for (bool zero_cond_t : {false, true}) {
        auto reference = run(false, argc == 2, zero_cond_t);
        auto split = run(true, argc == 2, zero_cond_t);
        if (reference.size() != split.size()) return 1;
        float max_error = 0.f;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(split[i])) return 1;
            max_error = std::max(max_error, std::abs(reference[i] - split[i]));
        }
        std::printf("Qwen edit%s whole vs sequence: max absolute error %.9g\n", zero_cond_t ? " zero-timestep refs" : "", max_error);
        if (max_error > 0.002f) return 2;
    }
    return 0;
}
