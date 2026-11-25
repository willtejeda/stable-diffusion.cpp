// Offline VAE-only encode/replay; never loads the diffusion or text model.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "model/vae/minimax_h3_vae.hpp"
#include "model/vae/tae.hpp"
#include "model_manager.h"

namespace {
    sd::Tensor<float> read_tensor(const char* path) {
        constexpr int32_t rank = 5;
        int32_t header[3 + rank];
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
        const auto size = file.tellg();
        file.seekg(0);
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        if (!file || header[0] != rank || header[1] != 0 || header[2] != GGML_TYPE_F32) {
            throw std::invalid_argument("expected a nameless, rank-five float32 tensor");
        }
        const auto payload = static_cast<int64_t>(size) - static_cast<int64_t>(sizeof(header));
        int64_t elements   = 1;
        std::vector<int64_t> shape;
        for (size_t i = 3; i < 3 + rank; ++i) {
            const int64_t extent = header[i];
            if (extent < 1 || elements > payload / static_cast<int64_t>(sizeof(float)) / extent) {
                throw std::invalid_argument("invalid tensor extent");
            }
            elements *= extent;
            shape.push_back(extent);
        }
        if (elements * static_cast<int64_t>(sizeof(float)) != payload) {
            throw std::invalid_argument("unexpected tensor payload length");
        }
        sd::Tensor<float> tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.data()), payload);
        if (!file) {
            throw std::runtime_error("could not read tensor payload");
        }
        return tensor;
    }

    void write_tensor(const char* path, const sd::Tensor<float>& tensor) {
        // Same storage order and header as SD_VIDEO_LATENT_EXPORT_FILE.
        int32_t header[8] = {5, 0, GGML_TYPE_F32};
        for (size_t i = 0; i < 5; ++i) {
            header[3 + i] = static_cast<int32_t>(tensor.shape()[i]);
        }
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary);
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        file.write(reinterpret_cast<const char*>(tensor.data()), tensor.numel() * sizeof(float));
        file.close();
        if (!file) {
            throw std::runtime_error("could not write output tensor");
        }
    }
}

int main(int argc, char** argv) try {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr, "usage: sd-video-vae <h3-encode|h3-full|h3-tae|wan-tae> <cpu|gpu0> <weights> <input.tensor> <output.tensor> <threads> [tiny-latent-tile]\n");
        return 2;
    }
    const std::string_view mode = argv[1];
    const bool encode = mode == "h3-encode";
    const bool wan = mode == "wan-tae";
    const bool tiny = wan || mode == "h3-tae";
    if (!encode && !tiny && mode != "h3-full") {
        throw std::invalid_argument("unknown VAE mode");
    }
    const std::string_view device = argv[2];
    if (device != "cpu" && device != "gpu0") {
        throw std::invalid_argument("device must be cpu or the single visible gpu0");
    }
    const int threads = std::stoi(argv[6]);
    const int tile = argc == 8 ? std::stoi(argv[7]) : 0;
    if (threads <= 0 || tile < 0 || (tile && !tiny)) {
        throw std::invalid_argument("invalid thread count or tiny decoder latent tile size");
    }
    auto input = read_tensor(argv[4]);
    if (input.dim() != 5 || input.shape()[3] != (encode ? 3 : wan ? 48 : 24) || input.shape()[4] != 1 ||
        (encode && (input.shape()[0] % 16 || input.shape()[1] % 16 || input.shape()[2] % 17)) ||
        (!encode && !wan && input.shape()[2] % 5 != 2)) {
        throw std::invalid_argument("unexpected clip dimensions or temporal grid");
    }
    sd_set_log_callback([](enum sd_log_level_t level, const char* text, void*) {
        if (level >= SD_LOG_WARN) {
            std::fputs(text, stderr);
        }
    }, nullptr);
    ggml_backend_load_all();
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_init_by_type(device == "cpu" ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU, nullptr),
        &ggml_backend_free);
    if (!backend) {
        throw std::runtime_error("requested backend unavailable");
    }
    ModelLoader loader;
    const auto version = wan ? VERSION_WAN2_2_TI2V : VERSION_MINIMAX_H3;
    if (!loader.init_from_file_and_convert_name(argv[3], tiny ? "tae." : "first_stage_model.", version)) {
        throw std::runtime_error("could not open VAE weights");
    }
    std::unique_ptr<VAE> runner;  // The manager releases tensor storage before the runner frees its context.
    auto manager = std::make_shared<ModelManager>();
    if (tiny) {
        runner = std::make_unique<TinyVideoAutoEncoder>(
            backend.get(), loader.get_tensor_storage_map(), "decoder", true, version, manager);
    } else {
        runner = std::make_unique<MiniMaxH3VAE::MiniMaxH3VideoVAERunner>(
            backend.get(), loader.get_tensor_storage_map(), "first_stage_model", manager);
    }
    manager->set_n_threads(threads);
    if (!manager->set_loader(std::move(loader)) ||
        !manager->register_runner_params(ModelComponent::VAE, *runner, ModelManager::ResidencyMode::ParamBackend,
                                         backend.get(), backend.get()) ||
        !manager->validate_registered_tensors() || !manager->load_all_params_eagerly()) {
        throw std::runtime_error("could not load VAE weights onto the requested backend");
    }
    runner->set_flash_attention_enabled(false);
    sd_tiling_params_t tiling = {};
    const std::string extra = "temporal_tile_frames=" + std::to_string(tile);
    if (tile) {
        tiling.temporal_tiling = true;
        tiling.extra_tiling_args = extra.c_str();
    }
    const auto start = std::chrono::steady_clock::now();
    auto output = encode ? runner->encode(threads, input, tiling) :
                           runner->decode(threads, runner->diffusion_to_vae_latents(input), tiling, true);
    ggml_backend_synchronize(backend.get());
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (output.empty()) {
        throw std::runtime_error("VAE graph did not produce frames");
    }
    if (encode) {
        output = runner->vae_to_diffusion_latents(output);
    }
    write_tensor(argv[5], output);
    std::printf("{\"seconds\":%.6f,\"frames\":%lld}\n", seconds, static_cast<long long>(output.shape()[2]));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "sd-video-vae: %s\n", error.what());
    return 1;
}
