# How to Use

## Download weights

- Download Qwen Image
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/diffusion_models
    - gguf: https://huggingface.co/QuantStack/Qwen-Image-GGUF/tree/main
- Download vae
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae
- Download qwen_2.5_vl 7b
    - safetensors: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders
    - gguf: https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-GGUF/tree/main

## Examples

```
.\bin\Release\sd-cli.exe --diffusion-model  ..\models\diffusion_models\qwen-image-Q8_0.gguf --vae ..\models\vae\qwen_image_vae.safetensors  --llm ..\models\text_encoders\Qwen2.5-VL-7B-Instruct-Q8_0.gguf  -p '一个穿着"QWEN"标志的T恤的中国美女正拿着黑色的马克笔面相镜头微笑。她身后的玻璃板上手写体写着 “一、Qwen-Image的技术路线： 探索视觉生成基础模型的极限，开创理解与生成一体化的未来。二、Qwen-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、精准图像编辑。支持文字编辑、物体增减、风格变换。三、Qwen-Image的未来愿景：赋能专业内容创作、助力生成式AI发展。”' --cfg-scale 2.5 --sampling-method euler -v --offload-to-cpu -H 1024 -W 1024 --flow-shift 3
```

<img alt="qwen example" src="../assets/qwen/example.png" />

## Two-GPU sequence split

For single-image text-to-image with batch one, use:

```text
--backend 'diffusion=cuda0&cuda1,te=cuda0,vae=cuda0' --params-backend te=cpu --auto-fit off --split-mode diffusion=seq
```

Image tokens and image RoPE rows are split after the input projection. The text stream is mirrored. Joint attention gathers FP16 image K/V before RoPE, keeps text before image in K/V order, and evaluates mirrored text queries separately from local image queries. The image output is gathered after the final projection. Both GPUs hold the full diffusion weights.

Reference-image edits and zero-condition timestep-indexed variants (including the current Qwen Edit preset) remain unsplit. Their ordinary execution path is preserved; this does not add sequence splitting for edits. Dual-stream attention has a tiny deterministic CPU/two-GPU numerical test; real-weight image and timing measurements currently cover Qwen Image 2.1 Turbo only. See [Qwen Image 2.1](qwen_image_2.1.md#two-gpu-sequence-split) for tests and measurement commands.




