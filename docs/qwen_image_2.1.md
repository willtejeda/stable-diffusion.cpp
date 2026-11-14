# How to Use

Qwen Image 2.1 supports text-to-image generation and image editing, using Qwen3-VL-8B as the text encoder and its own VAE.

## Download weights

- Download Qwen Image 2.1
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models
    - gguf: https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae
- Download Qwen3-VL-8B-Instruct
    - safetensors (BF16 or INT8 convrot): https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders
    - gguf: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
    - For image editing with a GGUF text encoder, also download `mmproj-Qwen3VL-8B-Instruct-F16.gguf` from the same repository and pass it with `--llm_vision`.

Use `qwen_image_2.1_vae_bf16.safetensors` with this model. The earlier Qwen Image and Wan 2.2 VAE weights are not interchangeable with the Qwen Image 2.1 VAE weights.

For faster encoding, decoding, or previews, see [TAEQI2.1](taesd.md#qwen-image-21-taeqi21).

## Examples

Run the following commands from the build directory. Use image dimensions divisible by 32. The resolution-dependent flow schedule is selected automatically.

### Text to image

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors --vae ..\models\vae\qwen_image_2.1_vae_bf16.safetensors --llm ..\models\text_encoders\Qwen3VL-8B-Instruct-Q4_K_M.gguf -p "a lovely cat holding a sign says 'qwen2.1.cpp'" --cfg-scale 6.0 --sampling-method euler -v --offload-to-cpu -o qwen_image_2.1.png
```

<img alt="Qwen Image 2.1 example" src="../assets/qwen/qwen_image_2.1.png" />

To use GGUF diffusion weights, set `--diffusion-model` to the path of a file such as `qwen_image_2.1-Q4_K.gguf`.

### Turbo

For [Qwen Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), use the Turbo diffusion weights and select its fixed 8-step sigma schedule with `--sigmas qwen-image-2.1-turbo`. For example, add these sampling options to the command above:

```text
--sigmas qwen-image-2.1-turbo --steps 8 --cfg-scale 1.0 --sampling-method euler
```

The preset is equivalent to `--sigmas 1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0`. It only supplies the sigma values; CFG scale and sampling method remain independently configurable. As with an explicit sigma list, the sigma count determines the actual number of sampling steps. The same preset name is also accepted by `--hires-sigmas` for the highres fix second pass.

### Two-GPU sequence split

For text-to-image, add these options (replace rather than duplicate existing backend options):

```text
--backend 'diffusion=cuda0&cuda1,te=cuda0,vae=cuda0' --params-backend te=cpu --auto-fit off --split-mode diffusion=seq
```

Weights are mirrored. Image tokens and their RoPE rows are split after the image input projection; text stays whole on each GPU. Each block gathers FP16 image K/V before RoPE and preserves text-first attention order. Text attention remains causal and cannot see the target image. The image output is gathered before unpatchifying. Both GPUs must fit the mirrored diffusion weights and their local working buffers.

Prefix caching is disabled on the sequence backend; the short text prefix is recomputed each step. Reference-image edits keep the full, unsplit graph. They are not an edit sequence-split implementation.

`test-qwen-sequence` checks separate-stream versus whole-stream attention and modulation with nonidentity RoPE, a causal text mask, and uneven image token counts. Its default run uses CPU; `--cuda` checks actual two-GPU collectives and token partitioning. Run that option only with the GPUs reserved. `tests/bench-qwen-sequence.py SIZE launcher|single|sequence` reproduces the local Turbo measurements and records commands, wall time, logs, and sampled VRAM under `build-h3seq`; it must be run inside a GPU lease.

#### Measured on the two P100s

Q6_K Turbo diffusion, Q4_K_M Qwen3-VL text encoder, original BF16 VAE file, Euler, fixed Turbo sigmas, CFG 1, eight steps, seed 42. Prompt: “A studio photograph of a red ceramic teapot with a curved handle and short spout, on a pale grey table, soft even lighting, three-quarter view, one object, no text.”

The read-only `run_sd.sh qwen21turbo --dry-run` preset assigns diffusion to CUDA1 and TE/VAE to CUDA0. It is a two-card **stage placement**, not a diffusion layer split. `launcher` below preserves those flags. `single` uses only physical GPU1; sequence uses the options above. All measurements used this worktree's `build-h3seq/bin/sd-cli`, not the installed server.

| Requested → actual size | Mode | Last denoising step s/it | Cold command wall s | Sampled peak MiB GPU0 / GPU1 | RGB PSNR vs launcher |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024 → 1024 | launcher | 5.93 | 74.48 | 13419 / 7665 | — |
| 1024 → 1024 | single | 6.07 | 77.20 | 0 / 14103 | — |
| 1024 → 1024 | sequence | 4.18 | 62.01 | 15159 / 7477 | 27.46 dB |
| 1328 → 1344 | launcher | 13.24 | 143.16 | 4693 / 8479 | — |
| 1328 → 1344 | single | 13.51 | 144.51 | 0 / 12787 | — |
| 1328 → 1344 | sequence | 7.98 | 106.39 | 8167 / 8167 | 42.50 dB |

These are single cold CLI runs, not repeated server throughput measurements. s/it is the logged eighth denoising step, not wall time divided by eight. Sampling totals (including first-step lazy weight loading) were 59.18 / 59.81 / 44.90 s at 1024 and 114.67 / 116.26 / 75.22 s at the larger size, in launcher / single / sequence order. The final sequence runs were remeasured after retaining unity-scale V in FP16 rather than redundantly widening and narrowing it. Both numerical GPU tests still passed (maximum absolute errors 0.00006134 dual-stream, 0.00001848 causal block).

The CLI rounded 1328 to actual 1344×1344 PNGs. All three larger runs rejected untiled VAE decoding for insufficient memory and automatically retried with spatial tiles; they completed successfully. Maximum sampled GPU temperature over the reported runs was 80 °C. VRAM is whole-command `nvidia-smi` sampling every 500 ms, including TE and VAE; it is not just diffusion memory. Automatic placement/retention can change those peaks.

PSNR was measured with `ffmpeg -lavfi psnr`; default RGBA averages were 28.71 and 43.74 dB. The table conservatively removes alpha with `format=gbrp` on both inputs (27.46 and 42.50 dB). Both images retain the same teapot scene.

Images: `/tmp/codex-shots/qwenseq-{1024,1328}-{launcher,single,sequence}.png`. Commands, seed, prompt, wall time, and sampled peaks are in `build-h3seq/qwenseq-measurements.jsonl` (last sequence entry per size is final). Per-case logs and VRAM CSVs use the same stems; `*-pre-v16*` artifacts retain the earlier sequence implementation's measurements. PSNR logs are `build-h3seq/qwenseq-{1024,1328}-{rgb-,}psnr.log`. No installed launcher, binary, model, or Hostler configuration was changed.


### Image editing

Pass the reference image with `-r` and describe the edit in `-p`. Vision weights are required; the example below loads them separately with `--llm_vision`.

```powershell
.\bin\Release\sd-cli.exe --diffusion-model ..\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors --vae ..\models\vae\qwen_image_2.1_vae_bf16.safetensors --llm ..\models\text_encoders\Qwen3VL-8B-Instruct-Q4_K_M.gguf --llm_vision ..\models\text_encoders\Qwen3VL-8B-Instruct-mmproj-BF16.gguf -r ..\assets\qwen\qwen_image_2.1.png -p "change 'qwen2.1.cpp' to 'sd.cpp'" --cfg-scale 6.0 --sampling-method euler -v --offload-to-cpu -o qwen_image_2.1_edit.png
```

For multiple reference images, repeat `-r` in the desired order, for example `-r first.png -r second.png`.

### Prefix cache

By default, the first denoising call for each fixed condition saves the text and reference-image keys and values from every transformer layer. Later calls only compute the target-image tokens. Positive and negative conditions use separate caches, which are released when sampling ends.

Set `qwen_image_2_1_prefix_cache_type` in `--model-args` to `auto` or a type name using the same parser and case-sensitive names as `--type`:

- `auto` (default): use FP16 only when Flash Attention is enabled, Sage Attention is disabled, the attention scale is unchanged, and every attention operation in the cache-writing or cache-reading graph selects Flash Attention after backend support checks. If an operation falls back, rebuild the prefix in FP32 before executing and keep FP32 for the rest of that sampling run.
- `f32`: always store FP32 keys and values.
- `f16`: always store FP16 keys and values, including with ordinary attention or custom attention scaling. This saves cache memory but can introduce additional rounding error.
- Other types, such as `bf16`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_K`, `q6_K`, `iq4_nl`, and `iq4_xs`: use the requested storage type if the ggml build provides runtime conversion to and from FP32. Quantization is lossy and must be selected explicitly; `auto` never selects a quantized type.

Cache data is packed into contiguous rows of `hidden_size` elements before conversion, so 256-element quantization blocks work with the model's 128-element attention heads without padding. The type's block size must divide `hidden_size`. Unknown types, types lacking runtime conversion (for example `q8_1` and several IQ formats), and incompatible block sizes are ignored with a warning, leaving the previous setting or the default `auto` unchanged.

For example, use `--model-args qwen_image_2_1_prefix_cache_type=q8_0` to enable 8-bit cache storage. Cached keys and values are converted back to the attention input type before concatenating with the current target tokens. This reduces persistent cache memory; attention working buffers still use floating-point values, and conversion adds work on each step. Backends without the required conversion operations use the existing CPU fallback.

For the default 32-layer model, a prefix of 4096 tokens takes approximately the following memory per condition, excluding weights, working buffers, and allocation overhead:

| Cache type | Memory |
| --- | ---: |
| `f32` | 4 GiB |
| `f16` | 2 GiB |
| `q8_0` | 1.0625 GiB |
| `q4_0` | 0.5625 GiB |

The runner accounts for the cache when checking the memory budget. If a cached execution runs out of memory, it releases the prefix caches, disables caching for the rest of that sampling run, and retries the full sequence once. Per-step conditioning extensions currently use the full-sequence path.

Disable this optimization with `--model-args qwen_image_2_1_prefix_cache=false`. It reuses step-independent activations; numerical results can still differ slightly because the matrix sizes change.

### Alpha channel

This model supports alpha channel output. As the model determines whether to output a regular image or with transparency through the prompt, according to [official recommendation](https://github.com/QwenLM/Qwen-Image-2.1#transparent-image-generation-rgba), use the following prompt format for better results:

> `This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent.`

Since transparency is decided by the prompt rather than by the input or an explicit switch, the same format applies equally to editing, whether or not the reference image itself has an alpha channel. Note that alpha is kept only in `.png` and `.webp` outputs; saving as `.jpg` drops the transparency.

Here are some examples ran with Q6_K quantization:
| Input | Prompt | Output |
| --- | --- | --- |
| ![Qwen Image 2.1 alpha input example 1](../assets/qwen/qwen-image-2.1-alpha-in1.png) | This is an RGBA image with transparency. Replace the text "BLOOM" with "Qwen Image 2.1", keeping the same font of the original text. The image has alpha channel and the background is transparent. | ![Qwen Image 2.1 alpha output example 1](../assets/qwen/qwen-image-2.1-alpha-out1.png) |
| ![Qwen Image 2.1 alpha input example 2](../assets/logo.png) | This is an RGBA image with transparency. Remove the background of the image, keeping only the text and cat. The image has alpha channel and the background is transparent. | ![Qwen Image 2.1 alpha output example 2](../assets/qwen/qwen-image-2.1-alpha-out2.png) |

### Other features

Other features of the model could be found on the [model card from QwenLM/Qwen-Image-2.1 repo](https://github.com/QwenLM/Qwen-Image-2.1), including 2 finetuned prompt rewriting Qwen3.5-9B model.
