#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "model/diffusion/flux3.hpp"

// Keep this test runnable before the sequence implementation is present.
template <typename T, typename = void>
struct has_sequence : std::false_type {};
template <typename T>
struct has_sequence<T, std::void_t<decltype(std::declval<T&>().forward_sequence(
    std::declval<GGMLRunnerContext*>(), std::declval<ggml_tensor*>(), std::declval<ggml_tensor*>(),
    std::declval<const Flux::ModulationOut&>(), std::declval<const Flux::ModulationOut&>(),
    std::declval<ggml_tensor*>(), std::declval<ggml_tensor*>(), std::declval<ggml_tensor*>()))>> : std::true_type {};

template <typename T>
static std::pair<ggml_tensor*, ggml_tensor*> sequence(
    T& block, GGMLRunnerContext* runner, ggml_tensor* img, ggml_tensor* txt,
    const Flux::ModulationOut& im, const Flux::ModulationOut& tm,
    ggml_tensor* local_pe, ggml_tensor* img_pe, ggml_tensor* txt_pe) {
    if constexpr (has_sequence<T>::value) {
        return block.forward_sequence(runner, img, txt, im, tm, local_pe, img_pe, txt_pe);
    } else {
        return {nullptr, nullptr};
    }
}

static std::vector<float> run(bool split, bool gpu) {
    auto single = gpu ? ggml_backend_init_by_name("CUDA0", nullptr) : sd_backend_cpu_init();
    GGML_ASSERT(single != nullptr);
    if (!gpu) sd_backend_cpu_set_n_threads(single, 1);
    SDTensorSplitContext split_context;
    split_context.sequence = true;
    auto backend = split && gpu ? sd_tensor_split_backend_init(
        {ggml_backend_get_device(single), ggml_backend_dev_by_name("CUDA1")}, &split_context) : single;
    for (auto card : sd_tensor_split_card_backends(backend)) sd_ggml_register_extensions(card);
    auto weights_ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto ctx = ggml_init({32 * 1024 * 1024, nullptr, true});
    constexpr int64_t dim = 256, head = 128, image_tokens = 7, text_tokens = 3;
    Flux3::Flux3Config config;
    config.hidden_size = dim;
    config.heads = dim / head;
    config.mlp_dim = 48;
    Flux3::Block img_block(config), txt_block(config), joint_block(config);
    std::map<std::string, ggml_tensor*> weights;
    for (auto entry : {std::make_pair(&img_block, "image."), std::make_pair(&txt_block, "text."),
                       std::make_pair(&joint_block, "joint.")}) {
        entry.first->init(weights_ctx);
        entry.first->get_param_tensors(weights, entry.second);
    }
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    int seed = 0;
    for (const auto& item : weights) {
        std::vector<float> values(ggml_nelements(item.second));
        for (auto& value : values) value = item.first.find("norm") != std::string::npos ? 1.f : 0.03f * std::sin(float(++seed));
        ggml_backend_tensor_set(item.second, values.data(), 0, values.size() * sizeof(float));
    }
    auto img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, image_tokens);
    auto txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, text_tokens);
    auto im_shift = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto im_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto im_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto txt_shift = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto txt_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto txt_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, head / 2, image_tokens + text_tokens);
    auto inputs_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    int input_seed = 0;
    for (auto tensor : {img, txt, im_shift, im_scale, im_gate, txt_shift, txt_scale, txt_gate}) {
        std::vector<float> values(ggml_nelements(tensor));
        for (auto& value : values) value = 0.4f * std::cos(float(++input_seed) * 0.17f);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    std::vector<std::vector<float>> positions;
    for (int i = 0; i < image_tokens + text_tokens; ++i) positions.push_back({0.f, float(i % 3), float(i), float(i % 5)});
    auto pe_values = Rope::embed_nd(positions, 1, 10000.f, {32, 32, 32, 32});
    ggml_backend_tensor_set(pe, pe_values.data(), 0, pe_values.size() * sizeof(float));
    Flux::ModulationOut im{im_shift, im_scale, im_gate}, tm{txt_shift, txt_scale, txt_gate};
    GGMLRunnerContext runner;
    runner.backend = backend;
    runner.ggml_ctx = ctx;
    runner.flash_attn_enabled = true;
    auto graph = ggml_new_graph_custom(ctx, 8192, false);
    runner.graph = graph;
    auto img_pe = ggml_ext_slice(ctx, pe, 3, 0, image_tokens);
    auto txt_pe = ggml_ext_slice(ctx, pe, 3, image_tokens, image_tokens + text_tokens);
    ggml_tensor *out_img, *out_txt;
    if (split) {
        img = sd_seq_split(ctx, backend, img, image_tokens);
        auto local_pe = sd_seq_split(ctx, backend, img_pe, image_tokens);
        img = sequence(img_block, &runner, img, nullptr, im, tm, local_pe, img_pe, nullptr).first;
        txt = txt_block.forward(&runner, txt, txt_pe, {tm}, {text_tokens});
        auto result = sequence(joint_block, &runner, img, txt, im, tm, local_pe, img_pe, txt_pe);
        out_img = sd_seq_gather(ctx, backend, result.first);
        out_txt = result.second;
    } else {
        img = img_block.forward(&runner, img, img_pe, {im}, {image_tokens});
        txt = txt_block.forward(&runner, txt, txt_pe, {tm}, {text_tokens});
        auto joint = ggml_concat(ctx, img, txt, 1);
        joint = joint_block.forward(&runner, joint, pe, {im, tm}, {image_tokens, text_tokens});
        out_img = Flux3::token_slice(ctx, joint, 0, image_tokens);
        out_txt = Flux3::token_slice(ctx, joint, image_tokens, text_tokens);
    }
    auto out = ggml_concat(ctx, out_img, out_txt, 1);
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
    if (!split || !gpu) ggml_backend_free(single);
    return result;
}

int main(int argc, char** argv) {
    if (!has_sequence<Flux3::Block>::value) {
        std::fprintf(stderr, "FLUX3 Block::forward_sequence is not implemented\n");
        return 1;
    }
    const bool gpu = argc == 2 && std::string(argv[1]) == "--cuda";
    auto reference = run(false, gpu);
    auto split = run(true, gpu);
    float max_error = 0.f;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(split[i])) return 1;
        max_error = std::max(max_error, std::abs(reference[i] - split[i]));
    }
    std::printf("FLUX3 early and joint blocks whole vs sequence: max absolute error %.9g\n", max_error);
    return max_error > 0.002f ? 2 : 0;
}
