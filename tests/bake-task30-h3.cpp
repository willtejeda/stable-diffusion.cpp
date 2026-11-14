// Task 30 offline bake: one CPU weight and its GPU LoRA delta at a time.
// Original tensor names, shapes and quantization types are retained.
#include <array>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>

#include "core/ggml_extend_backend.h"
#include "model/adapter/lora.hpp"
#include "model_io/gguf_io.h"
#include "name_conversion.h"

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static void export_weight(ggml_tensor* tensor, ggml_type type, uint8_t* output) {
    const int columns = static_cast<int>(tensor->ne[0]);
    const int rows = static_cast<int>(ggml_nelements(tensor) / columns);
    if (!ggml_is_quantized(type) || rows < 4) {
        convert_tensor(tensor->data, tensor->type, output, type, rows, columns);
        return;
    }
    require(tensor->type == GGML_TYPE_F32, "quantization requires a float source");
    const size_t stride = ggml_row_size(type, columns);
    std::vector<float> importance(columns, 1.0f);
    const auto convert_rows = [&](int worker) {
        const int begin = rows * worker / 4;
        const int count = rows * (worker + 1) / 4 - begin;
        const size_t written = ggml_quantize_chunk(type,
            static_cast<const float*>(tensor->data) + static_cast<size_t>(begin) * columns,
            output + static_cast<size_t>(begin) * stride, 0, count, columns, importance.data());
        require(written == static_cast<size_t>(count) * stride, "quantized row size mismatch");
    };
    std::array<std::future<void>, 3> workers;
    for (int worker = 0; worker < 3; ++worker)
        workers[worker] = std::async(std::launch::async, convert_rows, worker);
    convert_rows(3);
    for (auto& worker : workers) worker.get();
}

static void bake(const std::string& input, const std::string& adapter,
                 const std::string& output, bool cpu_only) {
    require(!std::filesystem::exists(output), "refusing to replace output");
    require(!std::filesystem::exists(output + ".part"), "partial output already exists");
    ModelLoader loader;
    require(loader.init_from_file(input), "cannot open source GGUF");
    ModelLoader merge_loader;
    require(merge_loader.init_from_file(input, "model.diffusion_model."), "cannot open diffusion namespace");
    const auto version = merge_loader.get_sd_version();
    const auto merge_name = [version](const std::string& name) {
        const auto prefixed = starts_with(name, "model.diffusion_model.") ? name : "model.diffusion_model." + name;
        return convert_tensor_name(prefixed, version);
    };
    const auto sources = loader.get_tensor_storage_map();
    require(!sources.empty(), "empty source model");
    std::vector<TensorWritePlan> plans;
    std::set<std::string> model_names;
    for (const auto& [name, source] : sources) {
        TensorWritePlan plan;
        plan.name = name;
        plan.type = source.type;
        plan.n_dims = source.n_dims;
        std::copy_n(source.ne, SD_MAX_DIMS, plan.ne);
        plans.push_back(plan);
        model_names.insert(merge_name(name));
    }
    ggml_backend_load_all();
    auto cpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), ggml_backend_free);
    auto gpu = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        cpu_only ? nullptr : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr), ggml_backend_free);
    require(cpu != nullptr && (cpu_only || gpu != nullptr), "requested backend unavailable");
    auto compute = cpu_only ? cpu.get() : gpu.get();
    sd_ggml_register_extensions(compute);
    auto manager = std::make_shared<ModelManager>();
    manager->set_n_threads(4);
    manager->set_enable_mmap(false);
    merge_loader.convert_tensors_name();
    require(manager->set_loader(std::move(merge_loader)), "cannot configure model manager");
    ModelManager::LoraSpec spec;
    spec.path = adapter;
    spec.required = true;
    std::vector<ModelManager::LoraSpec> specs{spec};
    require(manager->prepare_lora_sources(specs), "cannot register adapter");
    LoraModel lora("task30-bake", compute, cpu.get(), manager, specs.front().file_id,
                   version, ModelManager::ResidencyMode::Disk);
    require(lora.init_params(4), "cannot initialize adapter");
    GGUFStreamingWriter writer;
    std::string error;
    const auto partial = output + ".part";
    try {
        require(writer.write_metadata(partial, plans, &error), error);
        std::fstream file(partial, std::ios::binary | std::ios::in | std::ios::out);
        require(file.is_open(), "cannot open partial output");
        size_t index = 0, modified = 0;
        for (const auto& [name, source] : sources) {
            auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
                ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}), ggml_free);
            require(ctx != nullptr, "cannot allocate tensor metadata");
            // Expand quantized/BF16 weights to F32 before adding the adapter delta.
            auto load_type = ggml_is_quantized(source.type) || source.type == GGML_TYPE_BF16 ?
                             GGML_TYPE_F32 : source.type;
            auto tensor = ggml_new_tensor(ctx.get(), load_type, source.n_dims, source.ne);
            auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
                ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
            require(buffer != nullptr, "cannot allocate one source tensor");
            require(loader.load_tensor(source, tensor), "cannot load " + name);
            const auto merged_name = merge_name(name);
            require(lora.apply({{merged_name, tensor}}, model_names, version, 4, false), "adapter failed on " + name);
            std::vector<uint8_t> data(static_cast<size_t>(source.nbytes()));
            if (lora.applied_lora_tensors.empty()) {
                // Unmatched weights remain byte-identical, including quantization scales.
                std::ifstream original(input, std::ios::binary);
                original.seekg(static_cast<std::streamoff>(source.offset));
                original.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
                require(static_cast<size_t>(original.gcount()) == data.size(), "cannot copy original " + name);
            } else {
                export_weight(tensor, source.type, data.data());
                ++modified;
            }
            require(writer.write_tensor(file, index++, data.data(), data.size(), &error), error);
            std::cout << "BAKE " << index << '/' << sources.size() << ' ' << name
                      << " type=" << ggml_type_name(source.type)
                      << " modified=" << !lora.applied_lora_tensors.empty() << std::endl;
        }
        require(modified != 0, "adapter matched no source weights");
        file.flush();
        require(file.good(), "cannot flush baked model");
        file.close();
        require(std::filesystem::file_size(partial) == writer.file_size(), "baked file size mismatch");
        std::filesystem::rename(partial, output);
        std::cout << "BAKE_DONE tensors=" << sources.size() << " modified=" << modified << std::endl;
    } catch (...) {
        std::filesystem::remove(partial);
        throw;
    }
}

int main(int argc, char** argv) {
    if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--cpu")) {
        std::cerr << "usage: bake-task30-h3 SOURCE.gguf ADAPTER.gguf OUTPUT.gguf [--cpu]\n";
        return 2;
    }
    try {
        bake(argv[1], argv[2], argv[3], argc == 5);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
