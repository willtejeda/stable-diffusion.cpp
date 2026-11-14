#ifndef __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__

#include <algorithm>
#include <array>
#include <cinttypes>
#include "core/pe_cache.h"

#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace MOSSImage {
    struct MOSSImageConfig {
        int64_t dim      = 3072;
        int64_t ffn_dim  = 8192;
        int64_t text_dim = 4096;
        int64_t channels = 128;
        int64_t head_dim = 128;
        int64_t depth    = 42;
        int64_t freq_dim = 256;

        static MOSSImageConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            MOSSImageConfig config;
            int64_t depth = 0;
            for (const auto& [name, tensor] : weights) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                auto key = name.substr(prefix.size() + 1);
                if (key == "video_embedder.weight") {
                    config.channels = tensor.ne[0];
                    config.dim      = tensor.ne[1];
                } else if (key == "text_embedder.proj.weight") {
                    config.text_dim = tensor.ne[0];
                } else if (key == "layers.0.proj.ffn_down.weight") {
                    config.ffn_dim = tensor.ne[0];
                } else if (key == "layers.0.q_norm.weight") {
                    config.head_dim = tensor.ne[0];
                } else if (key == "time_embedding.0.weight") {
                    config.freq_dim = tensor.ne[0];
                }
                if (starts_with(key, "layers.")) {
                    depth = std::max<int64_t>(depth, std::stoi(key.substr(7)) + 1);
                }
            }
            if (depth) {
                config.depth = depth;
                LOG_VERBOSE("moss_image: dim = %" PRId64 ", ffn_dim = %" PRId64 ", depth = %" PRId64 ", head_dim = %" PRId64,
                            config.dim, config.ffn_dim, config.depth, config.head_dim);
            }
            return config;
        }
    };

    inline ggml_tensor* situ(ggml_context* c, ggml_tensor* hidden, ggml_tensor* gate) {
        auto a = ggml_scale(c, ggml_tanh(c, ggml_scale(c, gate, 0.25f)), 4.f);
        a      = ggml_mul(c, a, ggml_sigmoid(c, gate));
        auto b = ggml_scale(c, ggml_tanh(c, ggml_scale(c, hidden, 1.f / 25.f)), 25.f);
        return ggml_mul(c, a, b);
    }

    struct TextBlock : public GGMLBlock {
        MOSSImageConfig config;

        TextBlock(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]    = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["qkv"]      = std::make_shared<Linear>(config.dim, config.dim * 3, true);
            blocks["proj"]     = std::make_shared<Linear>(config.dim, config.dim, true);
            blocks["ffn_up"]   = std::make_shared<Linear>(config.dim, config.ffn_dim, true);
            blocks["ffn_gate"] = std::make_shared<Linear>(config.dim, config.ffn_dim, true);
            blocks["ffn_down"] = std::make_shared<Linear>(config.ffn_dim, config.dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto y      = norm("norm1", x);
            auto qkv    = linear("qkv", y);
            auto split  = [&](int i) {
                return ggml_reshape_4d(c, ggml_ext_slice(c, qkv, 0, i * config.dim, (i + 1) * config.dim), config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2]);
            };
            auto q = norm("q_norm", split(0));
            auto k = norm("k_norm", split(1));
            auto v = split(2);
            y      = Rope::attention(ctx, q, k, v, pe, nullptr, 1.f, false);
            y      = linear("proj", y);
            x      = ggml_add(c, x, y);
            y      = norm("norm2", x);
            y      = situ(c, linear("ffn_up", y), linear("ffn_gate", y));
            y      = linear("ffn_down", y);
            return ggml_add(c, x, y);
        }
    };

    struct Block : public GGMLBlock {
        MOSSImageConfig config;

        Block(const MOSSImageConfig& config)
            : config(config) {
            blocks["norm1"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["norm2"]            = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["q_norm"]           = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["k_norm"]           = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
            blocks["proj.qkv"]         = std::make_shared<Linear>(config.dim, config.dim * 3, false);
            blocks["proj.o"]           = std::make_shared<Linear>(config.dim, config.dim, false);
            blocks["proj.ffn_up_gate"] = std::make_shared<Linear>(config.dim, config.ffn_dim * 2, false);
            blocks["proj.ffn_down"]    = std::make_shared<Linear>(config.ffn_dim, config.dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, const std::array<ggml_tensor*, 6>& mod) {
            auto c      = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm   = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            auto part   = [&](int i) { return mod[i]; };
            auto y      = norm("norm1", x);
            y           = ggml_add(c, ggml_add(c, y, ggml_mul(c, y, part(1))), part(0));
            auto qkv    = linear("proj.qkv", y);
            auto split  = [&](int i) {
                return ggml_view_4d(c, qkv, config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2],
                                    config.head_dim * qkv->nb[0], qkv->nb[1], qkv->nb[2], i * config.dim * qkv->nb[0]);
            };
            auto q       = norm("q_norm", split(0));
            auto k       = norm("k_norm", split(1));
            auto v       = split(2);
            y            = Rope::attention(ctx, q, k, v, pe, nullptr, 1.f, false);
            y            = linear("proj.o", y);
            y            = ggml_mul(c, y, part(2));
            x            = ggml_add(c, x, y);
            y            = norm("norm2", x);
            y            = ggml_add(c, ggml_add(c, y, ggml_mul(c, y, part(4))), part(3));
            auto up_gate = linear("proj.ffn_up_gate", y);
            y            = situ(c, ggml_ext_slice(c, up_gate, 0, 0, config.ffn_dim), ggml_ext_slice(c, up_gate, 0, config.ffn_dim, 2 * config.ffn_dim));
            y            = linear("proj.ffn_down", y);
            y            = ggml_mul(c, y, part(5));
            return ggml_add(c, x, y);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward_sequence(
            GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* text,
            ggml_tensor* local_image_pe, ggml_tensor* image_pe, ggml_tensor* text_pe,
            const std::array<ggml_tensor*, 6>& mod) {
            auto c = ctx->ggml_ctx;
            auto linear = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            auto norm = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, h); };
            struct Stream {
                ggml_tensor *x, *q, *k, *v;
            };
            auto prepare = [&](ggml_tensor* x) {
                auto y = norm("norm1", x);
                y = ggml_add(c, ggml_add(c, y, ggml_mul(c, y, mod[1])), mod[0]);
                auto qkv = linear("proj.qkv", y);
                auto part = [&](int i) {
                    return ggml_view_4d(c, qkv, config.head_dim, config.dim / config.head_dim, x->ne[1], x->ne[2],
                                        config.head_dim * qkv->nb[0], qkv->nb[1], qkv->nb[2], i * config.dim * qkv->nb[0]);
                };
                return Stream{x, norm("q_norm", part(0)), norm("k_norm", part(1)), part(2)};
            };
            auto img = prepare(image);
            auto txt = prepare(text);
            // Exchange unrotated image K/V in FP16; text remains mirrored on both cards.
            auto img_k = ggml_cast(c, sd_seq_gather(c, ctx->backend, ggml_cast(c, img.k, GGML_TYPE_F16)), GGML_TYPE_F32);
            auto img_v = sd_seq_gather(c, ctx->backend, ggml_cast(c, img.v, GGML_TYPE_F16));
            auto txt_v = ggml_cast(c, txt.v, GGML_TYPE_F16);
            auto img_q = Rope::apply_rope(ctx, img.q, local_image_pe, false);
            img_k = Rope::apply_rope(ctx, img_k, image_pe, false);
            auto txt_q = Rope::apply_rope(ctx, txt.q, text_pe, false);
            auto txt_k = Rope::apply_rope(ctx, txt.k, text_pe, false);
            auto k = ggml_concat(c, img_k, txt_k, 1);
            auto v = ggml_concat(c, img_v, txt_v, 2);
            const auto heads = config.dim / config.head_dim;
            auto finish = [&](Stream& stream, ggml_tensor* q) {
                auto y = ggml_ext_attention_ext(ctx, q, k, v, heads, nullptr, true, ctx->flash_attn_enabled, 1.f);
                y = ggml_mul(c, linear("proj.o", y), mod[2]);
                auto x = ggml_add(c, stream.x, y);
                y = norm("norm2", x);
                y = ggml_add(c, ggml_add(c, y, ggml_mul(c, y, mod[4])), mod[3]);
                auto up_gate = linear("proj.ffn_up_gate", y);
                y = situ(c, ggml_ext_slice(c, up_gate, 0, 0, config.ffn_dim), ggml_ext_slice(c, up_gate, 0, config.ffn_dim, 2 * config.ffn_dim));
                y = ggml_mul(c, linear("proj.ffn_down", y), mod[5]);
                return ggml_add(c, x, y);
            };
            return {finish(img, img_q), finish(txt, txt_q)};
        }
    };

    struct Model : public GGMLBlock {
        MOSSImageConfig config;

        Model(const MOSSImageConfig& config)
            : config(config) {
            blocks["video_embedder"]     = std::make_shared<Linear>(config.channels, config.dim, true);
            blocks["text_embedder.proj"] = std::make_shared<Linear>(config.text_dim, config.dim, true);
            for (int i = 0; i < 2; ++i) {
                blocks["text_embedder.blocks." + std::to_string(i)] = std::make_shared<TextBlock>(config);
            }
            blocks["time_embedding.0"]        = std::make_shared<Linear>(config.freq_dim, config.dim, true);
            blocks["time_embedding.2"]        = std::make_shared<Linear>(config.dim, config.dim, true);
            blocks["shared_block_modulation"] = std::make_shared<Linear>(config.dim, 6 * config.dim, true);
            blocks["video_head_modulation"]   = std::make_shared<Linear>(config.dim, 2 * config.dim, true);
            for (int64_t i = 0; i < config.depth; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<Block>(config);
            }
            blocks["final_norm_video"]   = std::make_shared<RMSNorm>(config.dim, 1e-6f);
            blocks["final_linear_video"] = std::make_shared<Linear>(config.dim, config.channels, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* time, ggml_tensor* text, ggml_tensor* pe, const std::vector<int>& skip) {
            auto c            = ctx->ggml_ctx;
            auto linear       = [&](const char* name, ggml_tensor* h) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, h); };
            int64_t image_len = image->ne[0] * image->ne[1];
            auto x            = linear("video_embedder", DiT::patchify(c, image, 1, 1));
            text              = linear("text_embedder.proj", text);
            auto text_pe      = ggml_ext_slice(c, pe, 3, image_len, image_len + text->ne[1]);
            for (int i = 0; i < 2; ++i) {
                text = std::dynamic_pointer_cast<TextBlock>(blocks["text_embedder.blocks." + std::to_string(i)])->forward(ctx, text, text_pe);
            }
            time      = linear("time_embedding.2", ggml_silu(c, linear("time_embedding.0", time)));
            auto mod  = linear("shared_block_modulation", time);
            auto head = linear("video_head_modulation", time);
            std::array<ggml_tensor*, 6> parts;
            for (int i = 0; i < 6; ++i) {
                parts[i] = ggml_ext_slice(c, mod, 0, i * config.dim, (i + 1) * config.dim);
                if (ctx->graph != nullptr) {
                    ggml_build_forward_expand(ctx->graph, parts[i]);
                }
            }
            const bool sequence = sd_backend_is_sequence_split(ctx->backend) && x->ne[2] == 1 && image_len >= 2;
            ggml_tensor *image_pe = nullptr, *local_image_pe = nullptr;
            if (sequence) {
                image_pe = ggml_ext_slice(c, pe, 3, 0, image_len);
                local_image_pe = sd_seq_split(c, ctx->backend, image_pe, image_len);
                x = sd_seq_split(c, ctx->backend, x, image_len);
            } else {
                x = ggml_concat(c, x, text, 1);
            }
            for (int64_t i = 0; i < config.depth; ++i) {
                if (std::find(skip.begin(), skip.end(), i) == skip.end()) {
                    auto block = std::dynamic_pointer_cast<Block>(blocks["layers." + std::to_string(i)]);
                    if (sequence) {
                        auto out = block->forward_sequence(ctx, x, text, local_image_pe, image_pe, text_pe, parts);
                        x = out.first;
                        text = out.second;
                        sd::ggml_graph_cut::mark_graph_cut(text, "moss_image.layers." + std::to_string(i), "text");
                    } else {
                        x = block->forward(ctx, x, pe, parts);
                    }
                    sd::ggml_graph_cut::mark_graph_cut(x, "moss_image.layers." + std::to_string(i), "x");
                }
            }
            if (!sequence) {
                x = ggml_ext_slice(c, x, 1, 0, image_len);
            }
            x          = std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm_video"])->forward(ctx, x);
            auto shift = ggml_ext_slice(c, head, 0, 0, config.dim);
            auto scale = ggml_ext_slice(c, head, 0, config.dim, 2 * config.dim);
            x          = ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift);
            x          = linear("final_linear_video", x);
            if (sequence) {
                x = sd_seq_gather(c, ctx->backend, x);
            }
            return DiT::unpatchify(c, x, image->ne[1], image->ne[0], 1, 1);
        }
    };

    struct MOSSImageRunner : public DiffusionModelRunner {
        sd::PECache pe_cache_;
        MOSSImageConfig config;
        Model model;
        std::vector<float> time_data;

        MOSSImageRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager), config(MOSSImageConfig::detect_from_weights(weights, prefix)), model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "MOSS-Image"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            if (!inputs.x || !inputs.context || !inputs.timesteps || inputs.context->empty()) {
                LOG_ERROR("MOSS-Image requires latents, timesteps and text features");
                return {};
            }
            std::vector<int> skip;
            if (auto extra = std::get_if<SkipLayerDiffusionExtra>(&inputs.extra)) {
                if (extra->skip_layers) {
                    skip = *extra->skip_layers;
                }
            }
            auto graph = [&]() {
                auto gf   = new_graph_custom(65536);
                auto x    = make_input(*inputs.x);
                auto text = make_input(*inputs.context);
                GGML_ASSERT(x->ne[3] == 1 && x->ne[2] == config.channels && text->ne[0] == config.text_dim);
                int64_t h = x->ne[1], w = x->ne[0], nt = text->ne[1];
                sd::PECacheKey pe_key("moss_image.pe");
                pe_key.add(h).add(w).add(nt).add(config.head_dim).add(circular_x_enabled).add(circular_y_enabled);
                auto pe = make_input(pe_cache_.get_or_create(pe_key, [&]() {
                    Rope::Embedding embedding;
                    embedding.positions.append_image(static_cast<int>(h), static_cast<int>(w));
                    embedding.positions.images.back().height_axis = 2;
                    embedding.positions.images.back().width_axis  = 3;
                    embedding.positions.append_tokens(nt);
                    for (int64_t token = 0; token < h * w + nt; ++token) {
                        float pos[4] = {0.f, float(token - h * w), 0.f, 0.f};
                        if (token < h * w) {
                            pos[1] = float(nt);
                            pos[2] = float(token / w) - float(h - 1) / 2.f;
                            pos[3] = float(token % w) - float(w - 1) / 2.f;
                        }
                        embedding.ids.push_back({pos[0], pos[1], pos[2], pos[3]});
                    }
                    embedding.values = Rope::embed_nd(embedding.ids, 1, 10000.f, std::vector<int>(4, static_cast<int>(config.head_dim / 4)), embedding.layout, &embedding.frequencies);
                    auto values      = finish_rope_pe(std::move(embedding));
                    return sd::Tensor<float>({2, 2, config.head_dim / 2, h * w + nt}, std::move(values));
                }));
                time_data.resize(config.freq_dim);
                for (int64_t i = 0; i < config.freq_dim / 2; ++i) {
                    double angle                       = inputs.timesteps->values()[0] * std::pow(10000.0, -double(i) / double(config.freq_dim / 2));
                    time_data[i]                       = float(std::cos(angle));
                    time_data[i + config.freq_dim / 2] = float(std::sin(angle));
                }
                auto time = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, config.freq_dim, 1, 1);
                set_backend_tensor_data(time, time_data.data());
                auto ctx = get_context(gf);
                ggml_build_forward_expand(gf, model.forward(&ctx, x, time, text, pe, skip));
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), inputs.x->dim());
        }
    };
}  // namespace MOSSImage

#endif  // __SD_MODEL_DIFFUSION_MOSS_IMAGE_HPP__
