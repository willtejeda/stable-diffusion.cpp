#!/usr/bin/env python3
"""Run inside with_lease.sh; never acquires GPUs or changes Hostler itself."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build-h3seq"
MODELS = Path("/home/will/Inference/stable-diffusion.cpp/models/qwen21")
PROMPT = (
    "A studio photograph of a red ceramic teapot with a curved handle and short spout, "
    "on a pale grey table, soft even lighting, three-quarter view, one object, no text."
)


def run(size, mode):
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:")))
    if available < 20 * 1024 * 1024:
        raise RuntimeError("Less than 20 GiB RAM available")
    name = f"qwenseq-{size}-{mode}"
    image = Path("/tmp/codex-shots") / f"{name}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(OUT / "bin/sd-cli"),
        "--diffusion-model", str(MODELS / "qwen-image-2.1-turbo-Q6_K.gguf"),
        "--llm", str(MODELS / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        "--vae", str(MODELS / "qwen_image_2.1_vae_bf16.safetensors"),
        "--llm_vision", str(MODELS / "mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
        "--sampling-method", "euler", "--cfg-scale", "1.0", "--steps", "8",
        "-W", str(size), "-H", str(size), "--sigmas", "qwen-image-2.1-turbo",
        "--diffusion-fa", "--seed", "42", "--prompt", PROMPT, "--output", str(image),
    ]
    environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", GGML_CUDA_FA_FOLD="2")
    if mode == "launcher":
        command += ["--backend", "diffusion=cuda1,te=cuda0,vae=cuda0", "--auto-fit", "on"]
    elif mode == "single":
        environment["CUDA_VISIBLE_DEVICES"] = "1"
        command += ["--backend", "diffusion=cuda0,te=cuda0,vae=cuda0", "--auto-fit", "on"]
    elif mode == "sequence":
        command += ["--backend", "diffusion=cuda0&cuda1,te=cuda0,vae=cuda0", "--auto-fit", "off",
                    "--params-backend", "te=cpu", "--split-mode", "diffusion=seq"]
    else:
        raise ValueError(mode)
    record = {"size": size, "mode": mode, "command": command, "seed": 42, "prompt": PROMPT,
              "image": str(image), "env": {k: environment[k] for k in
              ("CUDA_DEVICE_ORDER", "GGML_CUDA_FA_FOLD", "CUDA_VISIBLE_DEVICES") if k in environment}}
    with (OUT / f"{name}-vram.csv").open("w") as samples, (OUT / f"{name}.log").open("w") as log:
        sampler = subprocess.Popen([
            "nvidia-smi", "--query-gpu=index,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits", "-lms", "500"], stdout=samples, stderr=subprocess.DEVNULL)
        start = time.monotonic()
        try:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=1700, check=False)
            record["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = "timeout"
        finally:
            record["wall_seconds"] = time.monotonic() - start
            sampler.terminate()
            sampler.wait()
    peaks = [0, 0]
    for line in (OUT / f"{name}-vram.csv").read_text().splitlines():
        fields = line.split(",")
        if len(fields) == 4:
            index, memory = map(int, fields[:2])
            peaks[index] = max(peaks[index], memory)
    record["peak_mib"] = peaks
    with (OUT / "qwenseq-measurements.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if record["returncode"] != 0:
        print((OUT / f"{name}.log").read_text()[-12000:], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage (under GPU lease): bench-qwen-sequence.py SIZE launcher|single|sequence")
    run(int(sys.argv[1]), sys.argv[2])
