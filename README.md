# Gemma 4 Multimodal Chat

A self-hosted FastAPI web service that wraps Google's **Gemma 4 27B** instruction-tuned model with a multimodal chat interface. It accepts text, images, and audio/video files in the same conversation, transcribes audio with OpenAI's **Whisper medium** model, and streams responses back to the browser with full Markdown rendering.

It runs locally on any Linux/Windows/macOS machine with a CUDA-capable GPU, and it ships with a SLURM batch script for running on an HPC cluster.

---

## Table of contents

1. [Features](#features)
2. [What it does NOT do](#what-it-does-not-do)
3. [Hardware requirements](#hardware-requirements)
4. [Software prerequisites](#software-prerequisites)
5. [Installation](#installation)
6. [Downloading the models](#downloading-the-models)
7. [Running it locally](#running-it-locally)
8. [Running it on a SLURM HPC cluster](#running-it-on-a-slurm-hpc-cluster)
9. [Using the chat UI](#using-the-chat-ui)
10. [Image generation: Flux.1 schnell](#image-generation-flux1-schnell)
11. [Video generation: Wan2.1 1.3B](#video-generation-wan21-13b)
12. [Configuration reference](#configuration-reference)
13. [API endpoints](#api-endpoints)
14. [How long audio is handled](#how-long-audio-is-handled)
15. [Troubleshooting](#troubleshooting)
16. [Project structure](#project-structure)

---

## Features

- **Multimodal chat** — send text, images, audio, or video files in any combination.
- **Streaming responses** — Gemma 4's tokens arrive in the browser as they are generated (Server-Sent Events).
- **Vision input** — drop any image (JPG/PNG/WebP/etc.) into the chat; Gemma 4 sees it directly.
- **Audio transcription** — any audio file (MP3, WAV, M4A, OGG, FLAC, AAC) is transcribed on the GPU with Whisper medium.
- **Video files** — video files (MP4, WebM, MOV) are accepted; the audio track is extracted by ffmpeg and transcribed. There is no visual frame analysis of videos (see *What it does NOT do*).
- **Chunked summarisation of long audio** — transcripts longer than 1 200 words are automatically split into ~900-word segments, each summarised individually, then a final answer is composed from the summaries. Lets you analyse 30+ minute podcasts on a small GPU.
- **Image generation** — `/imageflux <prompt>` generates a high-quality image (~85 sec) using **Flux.1 schnell**, the only open model that reliably renders readable text in images
- **Video generation — two modes** —
  - `/video <prompt>` — 5-second clip using **Wan2.1 1.3B** (~8 min on a 20 GB MIG slice). Real CFG guidance means the model follows the prompt.
  - `/videolora <lora> <prompt>` — same quality base but with a **LoRA from `lkzd7/WAN2.2_LoraSet_NSFW`** applied on top of **Wan2.2 T2V 14B** (~30 min). Available LoRAs: `doggy`, `spoon`, `sfbehind`, `transition`. The clip plays inline with standard browser controls.
- **GitHub-flavoured Markdown rendering** — headings, bullet/numbered lists, tables, blockquotes, code blocks, inline code, links, bold/italic. Sanitised with DOMPurify.
- **Dark-themed responsive UI** — looks clean on desktop and mobile.
- **Conversation history** — the browser keeps a rolling chat history (text only) and sends it back with each message for multi-turn context.
- **Live status updates** — the typing bubble shows what the model is doing (`Transcribing audio…`, `Summarising segment 3/7…`, etc.).
- **OOM-safe** — out-of-memory errors during inference are caught and surfaced as a readable message instead of silently failing.
- **Single-file deployment** — the whole frontend is inlined in `llm_chat_app.py`. There is nothing else to serve.

## What it does NOT do

- **No visual video understanding** — only the audio track of a video is used. Frames are not seen by the model.
- **No music or environmental sound recognition** — Whisper is a *speech-to-text* model. If you upload a song with no vocals, it will produce empty or hallucinated text. Use a fingerprinting service like AudD/ACRCloud for music ID.
- **No persistent conversations** — refresh the page and history is gone. There is no database.
- **No authentication / multi-user support** — anyone who can reach the port can use it. Bind to `127.0.0.1` or put it behind a tunnel/VPN.
- **No live microphone input** — only file uploads are supported.
- **No model switching at runtime** — the model is loaded once at startup. Change `MODEL_PATH` and restart to swap.
- **No conversation export** — history lives in the browser only.
- **No retry/regenerate button** — re-ask manually if a response is bad.
- **No image input from URLs** — uploads only.
- **One inference at a time** — a `threading.Lock` serialises generation. Concurrent requests will queue.
- **No image-to-image / inpainting** — only text-to-image generation. No ControlNet, no img2img, no editing of an uploaded image.
- **No NSFW filtering or safety classifier** — the image and video services pass the raw model output through. Don't expose this to untrusted users.
- **Gemma doesn't automatically call the image or video generator** — you have to type `/imageflux` or `/video` explicitly. There is no tool-calling that lets the model itself decide to generate media.

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU      | NVIDIA, ≥ 20 GB VRAM (e.g. A100 MIG slice, RTX 3090, A6000, H100) | A100 40 GB or H100 |
| System RAM | 32 GB | 64 GB |
| Disk    | ~20 GB free (15 GB Gemma 4 + 1.5 GB Whisper + ~3 GB env) | 40 GB |
| CPU     | 8 cores | 16 cores |
| OS      | Linux x86_64, CUDA 12.4 driver | Same |

Gemma 4 27B loaded in `bfloat16` occupies about **15 GB** of VRAM. Whisper medium on CUDA adds about **1.5 GB**. The remaining VRAM holds the KV cache during generation, which is why the chunked-summarisation mode is important for long transcripts.

If you enable the optional image-generation services, each one wants its **own GPU** (or its own MIG slice). They cannot share VRAM with the chat job:

| Service | Min VRAM | Disk weight cache |
|---|---|---|
| Flux.1 schnell | ~8 GB (with sequential offload) or ~24 GB (without) | ~24 GB |
| Wan2.1 1.3B (video) | ~8 GB | ~9 GB |

Total disk if you run everything: ~55 GB.

CPU-only inference is technically possible but extremely slow (multiple minutes per token) and is not recommended.

---

## Software prerequisites

- **Python 3.11** (other 3.10+ versions may work but were not tested)
- **NVIDIA driver** with CUDA 12.4 support — check with `nvidia-smi`
- **conda / miniconda** (recommended) or `venv`
- **ffmpeg** — required by Whisper for audio decoding. Install via conda or your system package manager.
- **git** — for cloning this repo

---

## Installation

The steps below assume Linux/macOS. On Windows, use Git Bash or WSL — paths and commands are otherwise identical.

### 1. Clone or copy the project

```bash
mkdir -p ~/llm_experiments
cd ~/llm_experiments
# Copy llm_chat_app.py (and serve_llm.slurm if using HPC) into this directory.
```

### 2. Create a conda environment

```bash
conda create -n rag_gemma4 python=3.11 -y
conda activate rag_gemma4
```

### 3. Install PyTorch with CUDA 12.4

PyTorch must be installed from the official wheel index — pip's default repo will give you the CPU-only build.

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

If `torchvision` later complains with `operator torchvision::nms does not exist`, force-reinstall it without touching torch:

```bash
pip install --force-reinstall --no-deps torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install the Python dependencies

```bash
pip install \
  transformers==4.50.0 \
  accelerate==1.13.0 \
  fastapi==0.136.1 \
  uvicorn==0.47.0 \
  python-multipart==0.0.29 \
  pillow==12.2.0 \
  openai-whisper==20250625 \
  huggingface_hub==1.15.0
```

> **Note on Transformers version:** Gemma 4 (`Gemma4ForConditionalGeneration`) requires Transformers ≥ 4.50. If your build is newer than 5.x, the API is the same — install whatever is current.

### 5. Install ffmpeg

```bash
conda install -c conda-forge ffmpeg -y
# or, system-wide:
#   Ubuntu/Debian:  sudo apt install ffmpeg
#   macOS:          brew install ffmpeg
```

Verify:

```bash
ffmpeg -version
```

### 6. Verify CUDA works inside the env

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You should see your GPU name printed. If `CUDA available: False`, your driver and the torch CUDA build don't match — re-check step 3.

---

## Downloading the models

### Gemma 4 27B

Gemma 4 is a gated model on both Kaggle and Hugging Face. You need to accept the licence agreement once.

**Option A — Hugging Face**

```bash
pip install -U huggingface_hub
huggingface-cli login        # paste your HF access token

huggingface-cli download google/gemma-4-27b-it \
  --local-dir /path/to/gemma4 \
  --local-dir-use-symlinks False
```

**Option B — Kaggle**

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle models instances versions download google/gemma-4/transformers/gemma-4-27b-it/1 \
  -p /path/to/gemma4 --unzip
```

Either way you'll end up with a directory containing `config.json`, `model-*.safetensors`, `tokenizer.model`, etc. Roughly **15 GB** on disk.

### Whisper medium

Whisper downloads automatically the first time you load it, but you can prefetch:

```bash
python -c "import whisper; whisper.load_model('medium', download_root='/path/to/whisper')"
```

This pulls a single ~1.5 GB `.pt` file into `/path/to/whisper`.

### Tell the app where the models are

The app reads two environment variables:

```bash
export MODEL_PATH=/path/to/gemma4
export WHISPER_PATH=/path/to/whisper
```

Defaults (used when the variables are unset) are `/scratch/users/t07an25/llm_experiments/gemma4` and `.../whisper`. Override them to match your machine.

---

## Running it locally

```bash
conda activate rag_gemma4
export MODEL_PATH=/path/to/gemma4
export WHISPER_PATH=/path/to/whisper
export PORT=8766          # optional, defaults to 8766

uvicorn llm_chat_app:app --host 0.0.0.0 --port 8766 --timeout-keep-alive 300
```

Or just run the script directly:

```bash
python llm_chat_app.py
```

Wait for these lines (they take about a minute):

```
[startup] Gemma 4 ready.
[startup] Whisper ready.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8766
```

Open `http://localhost:8766` in your browser. The header dot turns green once `/ready` confirms the model is loaded.

> **Security note:** `--host 0.0.0.0` exposes the port to anyone on your network. Use `--host 127.0.0.1` if you only want it accessible from the same machine, or run it behind an SSH tunnel (see below).

---

## Running it on a SLURM HPC cluster

A ready-to-use SLURM script (`serve_llm.slurm`) is included. It requests an A100 MIG slice (3g.20gb), 8 CPUs, 32 GB RAM, and a 24-hour wall time.

### 1. Edit the script for your cluster

Open `serve_llm.slurm` and update:

- `--partition`, `--gres` — match your site's GPU partition naming.
- Any `module load` lines — match the module names on your cluster.
- `conda activate rag_gemma4` — point at your env name.
- `MODEL_PATH` / `WHISPER_PATH` — point at where you put the model files.

### 2. Submit the job

```bash
mkdir -p ~/llm_experiments/logs
cd ~/llm_experiments
sbatch serve_llm.slurm
```

You'll get back `Submitted batch job <JOBID>`.

### 3. Check it's running

```bash
squeue -u $USER
cat logs/<JOBID>_chat.out
```

You'll see lines like:

```
MIG UUID: MIG-...
============================================================
  Gemma 4 Chat  |  Node: gpu02  |  Port: 8766
  ssh -L 8766:gpu02:8766 me@cluster.example.com -N
  http://localhost:8766
============================================================
```

### 4. Open an SSH tunnel from your laptop

Copy the `ssh -L ...` line from the log and run it on **your local machine** (not the cluster). Leave it running.

```bash
ssh -L 8766:gpu02:8766 me@cluster.example.com -N
```

### 5. Open the UI

Go to `http://localhost:8766` in your local browser. Traffic is routed through the tunnel to the GPU node.

### 6. Stopping the job

```bash
scancel <JOBID>
```

---

## Using the chat UI

| Action | How |
|-------|-----|
| Send a text message | Type → Enter |
| Insert a newline | Shift + Enter |
| Attach an image | Click the picture icon → pick an image file |
| Attach audio/video | Same picker, choose an audio or video file (MP3/WAV/M4A/MP4/WebM/OGG/FLAC/AAC) |
| Remove an attachment | Click the red ✕ on its preview thumbnail |
| Send | Click the paper-plane button or hit Enter |

When you attach audio:
1. The typing bubble shows **🎙️ Transcribing audio…**
2. Once Whisper finishes, a grey "Whisper transcript" bubble appears with the full text.
3. If the transcript is **≤ 1 200 words**, it goes straight to Gemma 4 with your question.
4. If it's **longer**, the bubble shows **🧩 Summarising segment N/M…** as each chunk is processed, then the final answer is streamed.

If you submit audio with **no typed message**, the app silently asks Gemma to "provide a comprehensive summary of this audio content."

---

## Image generation: Flux.1 schnell

The chat app generates images on demand by **proxying to a dedicated microservice** — `flux_gen_app.py`. It runs as its own SLURM job on its own MIG slice and exposes a FastAPI service on port 8768.

```
┌────────────────────┐   /imageflux   ┌──────────────────────┐
│  llm_chat_app.py   │ ─────────────▶ │  flux_gen_app.py     │  ← Flux.1 schnell
│  (port 8766)       │                │  (port 8768)         │     ~85 sec/image
└────────────────────┘                └──────────────────────┘
```

Flux.1 schnell is the only open model that reliably renders readable text in images and produces correct anatomy. It's a 12 B-parameter transformer that doesn't fit on a 20 GB MIG slice natively — sequential CPU offload keeps peak VRAM under ~8 GB at the cost of ~85 sec per image.

### Step 1 — Install the diffusers library

```bash
conda activate rag_gemma4
pip install -U diffusers
```

You want **diffusers ≥ 0.32** because anything older imports a constant (`FLAX_WEIGHTS_NAME`) that was removed in `transformers` 5.x. Also remove broken `bitsandbytes` if present:

```bash
pip show bitsandbytes >/dev/null 2>&1 && pip uninstall bitsandbytes -y
```

### Step 2 — Get a Hugging Face access token (one-off)

Flux.1 schnell is **gated** even though it's Apache 2.0:

1. Visit https://huggingface.co/black-forest-labs/FLUX.1-schnell and click **"Agree and access repository"**.
2. Create a read token at https://huggingface.co/settings/tokens and save it on the cluster:

```bash
mkdir -p ~/.huggingface
echo 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

The Flux SLURM script picks this up automatically as `HF_TOKEN`. Do this once, then forget about it.

### Step 3 — Submit the job

```bash
cd ~/llm_experiments
sbatch serve_flux_gen.slurm
tail -f logs/<JOBID>_flux.out
```

The first run downloads **~24 GB** (`black-forest-labs/FLUX.1-schnell`) into `$HF_HOME`. Subsequent restarts load from cache in ~20 seconds. When you see `[startup] Flux.1 schnell ready.` it's accepting requests.

### Step 4 — Tell the chat app where the service lives

| Variable | Default | Purpose |
|---|---|---|
| `FLUX_GEN_URL` | `http://gpu02:8768` | URL of the Flux.1 schnell service |

If the job lands on a different node, add to `serve_llm.slurm`:

```bash
export FLUX_GEN_URL=http://<actual_node>:8768
```

### Step 5 — Use it from the chat UI

```
/imageflux a chalkboard with the words "Hello World" written in cursive
```

You'll see **🎨 Generating with Flux: …** in the status bubble, then the image inline with the prompt as caption.

### Step 6 — Test it directly (optional)

```bash
curl http://gpu02:8768/ready
# → {"ready":true}

curl -X POST http://gpu02:8768/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a sign saying ABC in chunky 3D letters"}' \
  | python -c "import sys,json,base64; d=json.load(sys.stdin); open('flux.png','wb').write(base64.b64decode(d['image'].split(',')[1])); print('saved flux.png')"
```

### Step 7 — Stopping the service

```bash
for j in $(squeue -u $USER -h -n flux_gen -o %i); do scancel $j; done
```

### Quick reference

| Task | Command |
|---|---|
| Install diffusers | `pip install -U diffusers` |
| Remove broken bitsandbytes | `pip uninstall bitsandbytes -y` |
| Save HF token | `echo hf_xxx > ~/.huggingface/token && chmod 600 ~/.huggingface/token` |
| Start Flux | `sbatch serve_flux_gen.slurm` |
| Test Flux | `curl http://gpu02:8768/ready` |
| Use from chat | `/imageflux <prompt>` |
| Stop the job | `scancel <JOBID>` |

---

## Video generation: Wan2.1 1.3B

The chat app can generate short video clips on demand by **proxying to a dedicated video-generation microservice** — `video_gen_app.py`. It runs as its own SLURM job on its own MIG slice and exposes a FastAPI service on port 8769.

```
┌────────────────────┐      /video      ┌──────────────────────────┐
│  llm_chat_app.py   │ ───────────────▶ │  video_gen_app.py        │  ← Wan2.1 1.3B
│  (port 8766)       │                  │  (port 8769)             │     ~7-8 min / 5s clip
└────────────────────┘                  └──────────────────────────┘
```

**Why Wan2.1 1.3B?**

Wan2.1 supports real classifier-free guidance (`guidance_scale=5.0`), meaning the model genuinely follows your prompt. Distilled models like LTX-Video 2B are locked to `guidance_scale=1.0` — they effectively ignore the prompt for multi-object or compositionally complex scenes. Wan2.1 is only 1.3 B parameters but produces dramatically better results for a wider range of subjects.

| Property | Value |
|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Resolution | 832 × 480 (480P widescreen) |
| Duration | 5 seconds (81 frames @ 16 fps) |
| Inference steps | 50 |
| Guidance scale | 5.0 (real CFG) |
| VRAM footprint | ~8 GB (`enable_model_cpu_offload`) |
| Generation time | ~7–8 min on an A100 MIG 20 GB slice |
| Output format | Base64-encoded MP4, played inline in the browser |

### Step 1 — Install imageio-ffmpeg

Wan2.1 uses `diffusers.utils.export_to_video` to assemble frames into an MP4. That function needs either `av` (PyAV) or `imageio + imageio-ffmpeg`. The `av` package requires system ffmpeg libraries to compile, so the simpler path is:

```bash
conda activate rag_gemma4
pip install imageio imageio-ffmpeg
```

No model download at this step — the weights are pulled automatically from Hugging Face on first run.

### Step 2 — Submit the video-gen job

```bash
cd ~/llm_experiments
sbatch serve_video_gen.slurm
```

Check the log:

```bash
tail -f logs/<JOBID>_video.out
```

The first run downloads **~9 GB** from Hugging Face (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) into `$HF_HOME`. Subsequent restarts load from cache in about 90 seconds. When you see:

```
[startup] Wan2.1 1.3B ready.
```

the service is ready to accept requests.

### Step 3 — Tell the chat app where the video service lives

The chat app reads one env var:

| Variable | Default | Purpose |
|---|---|---|
| `VIDEO_GEN_URL` | `http://gpu02:8769` | URL of the Wan2.1 video-generation service |

If the video job lands on a different node (check `squeue`), add to `serve_llm.slurm`:

```bash
export VIDEO_GEN_URL=http://<actual_node>:8769
```

…and restart the chat job.

### Step 4 — Test the video service directly (optional)

From the HPC head node (cluster's internal network):

```bash
curl http://gpu02:8769/ready
# → {"status":"ready"}

curl -X POST http://gpu02:8769/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a golden retriever running on a beach at sunset"}' \
  --max-time 600 \
  | python -c "
import sys, json, base64
d = json.load(sys.stdin)
open('test.mp4', 'wb').write(base64.b64decode(d['video']))
print(f'Saved test.mp4  ({d[\"num_frames\"]} frames @ {d[\"fps\"]} fps)')
"
```

### Step 5 — Use it from the chat UI

In the chat box, type:

```
/video a cat sitting on a rooftop watching city lights at night
```

You'll see a status bubble (**🎬 Generating video with Wan2.1 1.3B: …**), then the resulting video embedded inline with playback controls. The clip autoplays, loops silently, and you can unmute or go fullscreen with the standard browser controls.

Generation takes 7–8 minutes for a 5-second clip. If the job runs out of VRAM at 81 frames it automatically retries at 33 frames (~2 seconds) and adds a note to the response.

### Step 6 — Stopping the video-gen job

```bash
squeue -u $USER
scancel <JOBID>
# or by name:
for j in $(squeue -u $USER -h -n wan_video -o %i); do scancel $j; done
```

### Quick reference

| Task | Command |
|---|---|
| Install video deps | `pip install imageio imageio-ffmpeg` |
| Start video service | `sbatch serve_video_gen.slurm` |
| Check readiness | `curl http://gpu02:8769/ready` |
| Use from chat | `/video <prompt>` |
| Check GPU usage | `ssh gpu0X nvidia-smi` |
| Stop the job | `scancel <JOBID>` |

---

## LoRA video generation: Wan2.2 T2V 14B

A second video service (`video_lora_app.py`) runs the larger **Wan2.2 T2V A14B** model with LoRAs from [`lkzd7/WAN2.2_LoraSet_NSFW`](https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW). It runs on port 8770 as a separate SLURM job on its own MIG slice.

```
┌────────────────────┐   /videolora   ┌──────────────────────────────┐
│  llm_chat_app.py   │ ─────────────▶ │  video_lora_app.py           │  ← Wan2.2 14B
│  (port 8766)       │                │  (port 8770)                 │     + LoRA
└────────────────────┘                └──────────────────────────────┘
```

**Why a separate service?**

Wan2.2 T2V A14B is 14 B parameters (~28 GB bfloat16). It won't fit alongside Wan2.1 in the same process. With `enable_sequential_cpu_offload()` it fits in a 20 GB MIG slice but inference is slower: about **25–45 minutes per 5-second clip at 30 steps**.

**Why Wan 2.2 uses LoRA pairs (HIGH + LOW noise)**

Wan 2.2 has two separate denoising transformers. Each LoRA in the repo comes as two files:

| File suffix | Loaded into |
|---|---|
| `*_high_noise.safetensors` / `*_H.safetensors` | `transformer` (first denoiser) |
| `*_low_noise.safetensors` / `*_L.safetensors` | `transformer_2` (second denoiser, `load_into_transformer_2=True`) |

The service loads both halves automatically when you specify a LoRA name.

### Available LoRAs

| Name | Files |
|---|---|
| `doggy` | `mql_doggy_a_wan22_t2v_v1_{high,low}_noise.safetensors` |
| `spoon` | `mqlspn_a_wan22_t2v_v1_{high,low}_noise.safetensors` |
| `sfbehind` | `sfbehind_v2.1_{high,low}_noise.safetensors` |
| `transition` | `sid3l3g_transition_v2.0_{H,L}.safetensors` |

Only T2V (text-to-video) LoRAs are listed. The I2V LoRAs in the same repo require `WanImageToVideoPipeline` and are not supported by this service.

### Step 1 — Submit the LoRA service job

```bash
cd ~/llm_experiments
sbatch serve_video_lora.slurm
tail -f logs/<JOBID>_videolora.out
```

The first run downloads the **Wan2.2 T2V A14B model (~28 GB)** plus any requested LoRA files (~300–600 MB each) into `$HF_HOME`. Subsequent restarts load from cache in ~5 minutes. When you see:

```
[startup] Wan2.2 T2V 14B ready.
```

the service is accepting requests.

### Step 2 — Tell the chat app where the LoRA service lives

| Variable | Default | Purpose |
|---|---|---|
| `VIDEO_LORA_URL` | `http://gpu02:8770` | URL of the Wan2.2 LoRA video service |

If the job lands on a different node, add to `serve_llm.slurm`:

```bash
export VIDEO_LORA_URL=http://<actual_node>:8770
```

### Step 3 — Test it directly (optional)

```bash
curl http://gpu02:8770/loras
# → ["doggy","spoon","sfbehind","transition"]

curl http://gpu02:8770/ready
# → {"status":"ready"}

curl -X POST http://gpu02:8770/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"two people in a cozy bedroom at night","lora":"doggy"}' \
  --max-time 2700 \
  | python -c "
import sys, json, base64
d = json.load(sys.stdin)
open('lora_test.mp4', 'wb').write(base64.b64decode(d['video']))
print(f'Saved lora_test.mp4  (lora={d[\"lora\"]}, {d[\"num_frames\"]} frames)')
"
```

### Step 4 — Use from the chat UI

```
/videolora doggy two people in a cozy bedroom, cinematic lighting
```

The first word after `/videolora` is the LoRA name; everything else is the prompt. You'll see **🎬 Generating with Wan2.2·doggy: …** in the status bubble. The clip appears inline when done.

The currently loaded LoRA is cached on the service — switching to a different LoRA takes an extra ~30 s to swap the weights.

### Step 5 — Stopping the job

```bash
for j in $(squeue -u $USER -h -n wan_lora -o %i); do scancel $j; done
```

### Quick reference

| Task | Command |
|---|---|
| Start LoRA service | `sbatch serve_video_lora.slurm` |
| List available LoRAs | `curl http://gpu02:8770/loras` |
| Check readiness | `curl http://gpu02:8770/ready` |
| Use from chat | `/videolora <lora_name> <prompt>` |
| Stop the job | `scancel <JOBID>` |

---

## Configuration reference

All configuration is via environment variables:

| Variable | Default | Used by | Purpose |
|----------|---------|---------|---------|
| `MODEL_PATH` | `/scratch/users/t07an25/llm_experiments/gemma4` | chat | Directory holding the Gemma 4 model files |
| `WHISPER_PATH` | `/scratch/users/t07an25/llm_experiments/whisper` | chat | Directory holding the Whisper `.pt` file |
| `PORT` | `8766` / `8767` / `8768` | each service | HTTP port |
| `SYSTEM_PROMPT` | built-in default | chat | Prepended to every conversation |
| `FLUX_GEN_URL` | `http://gpu02:8768` | chat | Where to find the Flux.1 schnell service |
| `VIDEO_GEN_URL`  | `http://gpu02:8769` | chat | Where to find the Wan2.1 1.3B video service |
| `VIDEO_LORA_URL` | `http://gpu02:8770` | chat | Where to find the Wan2.2 14B LoRA video service |
| `VIDEO_LORA_MODEL` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | video LoRA service | Override the Wan2.2 base model |
| `VIDEO_LORA_REPO`  | `lkzd7/WAN2.2_LoraSet_NSFW` | video LoRA service | HF repo containing the LoRA safetensors |
| `HF_HOME` | `/scratch/users/t07an25/llm_experiments/hf_cache` | image/video services | Where to cache the diffusion model weights |
| `HF_TOKEN` | from `~/.huggingface/token` | Flux | HF access token for the gated Flux repo |
| `SDXL_BASE` | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL service | Override SDXL base model |
| `FLUX_MODEL` | `black-forest-labs/FLUX.1-schnell` | Flux service | Override Flux model variant |

Tunable constants live near the top of each file:

| Constant | File | Default | Purpose |
|----------|------|---------|---------|
| `CHUNK_WORDS` | `llm_chat_app.py` | `900` | Words per chunk in chunked summarisation mode |
| `LONG_TRANSCRIPT_WORDS` | `llm_chat_app.py` | `1200` | Threshold above which a transcript is chunked |
| `MAX_HISTORY_TURNS` | `llm_chat_app.py` | `6` | Last N user/model turns kept in context |
| `MAX_IMAGE_EDGE` | `llm_chat_app.py` | `896` | Downscale uploaded images so longest edge ≤ this |
| `MAX_INPUT_TOKENS_SOFT` | `llm_chat_app.py` | `6000` | If prompt exceeds this after history trim, drop more history |
| `N_STEPS` | `flux_gen_app.py` | `4` | Flux schnell is a 1–4 step distillation |
| `VIDEO_MODEL` | `video_gen_app.py` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Override the Wan2.1 model variant (env var) |

The `_model.generate(...)` call uses `max_new_tokens=1024`, `temperature=0.7`, `top_p=0.9`, `do_sample=True` — change these in the source if you want different sampling behaviour. Chunk summaries use `do_sample=False` (greedy) with `max_new_tokens=220` for stable, deterministic summaries.

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serves the chat UI (single HTML page) |
| `GET` | `/ready` | Returns `{"ready": true}` once Gemma 4 has finished loading |
| `POST` | `/chat` | Accepts a multipart form and streams the response as SSE |

### `POST /chat`

**Form fields (all optional except at least one of `message` / `image` / `audio`):**

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User text |
| `history` | string | JSON array of `{role, content}` objects from the previous turns |
| `image` | file | An image to send to Gemma 4 |
| `audio` | file | An audio or video file to transcribe and analyse |

**Response:** `text/event-stream`. Each event is a JSON object on a `data:` line:

| Field | Meaning |
|-------|---------|
| `{"status": "..."}` | Live progress update for the typing bubble (transcribing, summarising, generating image, …) |
| `{"transcript": "..."}` | Whisper's output, shown to the user as a separate bubble |
| `{"text": "..."}` | A generation chunk to append to the assistant's response |
| `{"generated_image": "data:image/png;base64,...", "prompt": "...", "model": "..."}` | A generated image to embed in the chat (from `/imageflux`) |
| `{"generated_video": "<base64 MP4>", "prompt": "...", "model": "...", "num_frames": N, "fps": 16}` | A generated video to embed in the chat (from `/video`) |
| `{"error": "..."}` | Something went wrong; the UI shows it as an error bubble |
| `{"done": true}` | End of stream |

#### Image-generation and video-generation services

Each microservice exposes the same two endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/ready` | Returns `{"ready": true}` once the model is loaded |
| `POST` | `/generate` | Generates one image |

**POST `/generate` body (JSON):**

```json
{
  "prompt": "a samurai cat wielding katanas, anime style",
  "negative_prompt": "blurry, low quality",   // SDXL only, optional
  "width": 1024,                                // optional, default 1024
  "height": 1024,                               // optional, default 1024
  "seed": 42                                    // optional, for reproducibility
}
```

**Response:**

```json
{
  "image": "data:image/png;base64,iVBORw0KG...",
  "prompt": "a samurai cat wielding katanas, anime style"
}
```

Or on failure:

```json
{ "error": "GPU OOM: ..." }
```

#### Video-generation service

`video_gen_app.py` exposes two endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/ready` | Returns `{"status": "ready"}` once the model is loaded |
| `POST` | `/generate` | Generates one video clip |

**POST `/generate` body (JSON):**

```json
{
  "prompt": "a fox running through a snowy forest",
  "negative_prompt": "worst quality, blurry, jittery, distorted",
  "width": 832,
  "height": 480,
  "num_frames": 81,
  "num_inference_steps": 50,
  "guidance_scale": 5.0,
  "seed": 42
}
```

All fields except `prompt` are optional.

**Response:**

```json
{
  "video": "<base64-encoded MP4>",
  "num_frames": 81,
  "fps": 16,
  "prompt": "a fox running through a snowy forest"
}
```

On OOM the service automatically retries with `num_frames=33` and adds `"note": "OOM on first attempt; fell back to 33 frames."` to the response.

---

## How long audio is handled

A naïve approach — passing a 30-minute transcript (~6 000 words) to Gemma 4 in one shot — easily exhausts VRAM on a 20 GB GPU because the KV cache scales with input length.

To avoid OOM **without truncating** the audio, this app does the following whenever the transcript is over `LONG_TRANSCRIPT_WORDS` (1 200) words:

1. Split the transcript into chunks of `CHUNK_WORDS` (900) words each.
2. For each chunk, run a fast greedy generation asking Gemma 4 to "concisely summarise this transcript segment". Cap at 220 new tokens.
3. Emit a `status` event to the browser between chunks: `🧩 Summarising segment N/M…`.
4. Concatenate the per-chunk summaries into a single context string of the form `Segment 1/7: …\n\nSegment 2/7: …`.
5. Build the final prompt as that combined summary + the user's question (or a default "provide a comprehensive summary" if no question was typed).
6. Stream the final answer normally.

The benefit: full transcript is preserved, the user sees it in the chat, but Gemma 4 only ever processes ~900 words at a time. The cost: an extra ~3–5 seconds per chunk.

---

## Troubleshooting

**`operator torchvision::nms does not exist`**
Mismatched torch / torchvision builds. Force-reinstall the matching torchvision wheel:
```bash
pip install --force-reinstall --no-deps torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

**`Gemma4VideoProcessor requires the Torchvision library`**
Torchvision is missing entirely. Install it (step 3 above).

**`[Transcription failed: [Errno 2] No such file or directory: 'ffmpeg']`**
ffmpeg isn't on the `PATH`. Install it (step 5 above) and restart the server.

**`CUDA out of memory` during a Gemma response**
Your GPU is too small for the input. For audio this should now be impossible because of chunked summarisation, but it can still happen with very large images. The error is now reported as a chat message instead of silently dying. Try a smaller image, lower `max_new_tokens`, or use a bigger GPU.

**Browser shows "Connection error: failed to fetch" mid-stream**
The SSH tunnel dropped. Re-run the `ssh -L ...` command. If the SLURM job itself was restarted, you may need to update the node name (`gpu02` → whatever the new job is on) — check `squeue` and the new log file.

**The page loads but the green dot never appears**
The model is still loading. Look at `logs/<JOBID>_chat.out` — you should see `[startup] Gemma 4 ready.` after about a minute. If you see a traceback instead, fix the underlying issue (usually a missing model file or a CUDA driver mismatch).

**The model takes forever to download**
Gemma 4 is 15 GB. On a slow link this can take a while. Use `huggingface-cli` with the `--max-workers 4` flag or run it inside `tmux` so it survives disconnects.

**The transcript is gibberish or wrong language**
Whisper auto-detects language but mis-detects sometimes (e.g. low-volume background music, non-speech audio). Whisper does not transcribe music with no vocals — it will hallucinate. There is no fix in this app; that is a Whisper limitation.

**`cannot import name 'FLAX_WEIGHTS_NAME' from 'transformers.utils'`** (image-gen services)
You have an older `diffusers` (≤ 0.31) paired with a newer `transformers` (≥ 5.x). Upgrade:
```bash
pip install -U diffusers
```

**`CUDA Setup failed despite GPU being available` / `bitsandbytes` error at import**
Broken `bitsandbytes` is being imported transitively by `diffusers`. Just remove it:
```bash
pip uninstall bitsandbytes -y
```

**`401 Client Error … Cannot access gated repo … FLUX.1-schnell`**
Flux is gated. Visit https://huggingface.co/black-forest-labs/FLUX.1-schnell and click "Agree", then save a read-only token from https://huggingface.co/settings/tokens to `~/.huggingface/token`. The Flux SLURM script picks it up automatically.

**`/imageflux` works but returns `GPU OOM` at the start of every request**
`enable_model_cpu_offload()` was chosen instead of `enable_sequential_cpu_offload()` — the full Flux transformer doesn't fit on a 20 GB slice. Open `flux_gen_app.py` and use `enable_sequential_cpu_offload()`. Slower, but actually fits.

**`Flux service unreachable at http://gpu02:8768`**
The flux_gen SLURM job isn't running. Submit it with `sbatch serve_flux_gen.slurm`.

**`/video` returns "Video service unreachable at http://gpu02:8769"`**
The video-gen SLURM job isn't running. Check `squeue -u $USER` for a `wan_video` job. Submit it with `sbatch serve_video_gen.slurm`.

**`/video` times out after a long wait**
Wan2.1 at 50 inference steps takes ~7–8 min per clip. The chat app's video timeout is 900 s. If you're consistently hitting it, reduce `num_inference_steps` to 30 in `video_gen_app.py` (quality trade-off: noticeable but acceptable).

**`export_to_video` fails with `No module named 'imageio'`**
Install the imageio backend: `pip install imageio imageio-ffmpeg`.

**Video output is a corrupted file / browser shows a broken video player**
This can happen if the video job OOMed mid-frame and returned partial data. Check the video job log (`tail logs/<JOBID>_video.out`) for an OOM traceback. The OOM retry (33 frames) should prevent this, but if the retry also OOMed, you'll need a bigger MIG slice or to lower `num_frames` in `GenRequest`.

**Generated video ignores the prompt / output looks like random noise**
Make sure the job is running the Wan2.1 service (`video_gen_app.py`), not an older LTX-Video version. The `guidance_scale=5.0` default in Wan2.1 is what makes the model follow prompts — confirm this in the request by checking the job log.

**`ModuleNotFoundError: No module named 'diffusers'` in video-gen log**
The video job didn't activate the conda env. Edit `serve_video_gen.slurm` and check the `conda activate rag_gemma4` line runs before `uvicorn`.

**`/videolora` returns "Unknown LoRA '…'"**
The LoRA name you typed doesn't match any key in `LORAS` inside `video_lora_app.py`. Available names: `doggy`, `spoon`, `sfbehind`, `transition`. Use `curl http://gpu02:8770/loras` to get the current list from the running service.

**`/videolora` service runs out of memory even with sequential offload**
The Wan2.2 14B model requires ~48 GB of CPU RAM when using sequential offload (model weights live in system RAM). If the SLURM job is OOM-killed, check `squeue`/`sacct` and increase `--mem` in `serve_video_lora.slurm` (current default: 48 G).

**LoRA switch takes a long time between `/videolora` calls**
Each LoRA switch calls `unload_lora_weights()` + two `load_lora_weights()` calls (HIGH + LOW), which download and re-patch the model. After the first use of each LoRA the files are cached in `$HF_HOME`, so subsequent switches take ~30 s rather than minutes.

**`load_into_transformer_2` keyword argument not recognised**
Your diffusers version is too old. This parameter was added for Wan 2.2 dual-denoiser support. Upgrade: `pip install -U diffusers`.

**`/videolora` times out (1800 s)**
Wan2.2 14B with sequential offload at 30 steps takes ~25–45 min per clip. Reduce `num_inference_steps` to 20 in `video_lora_app.py` or lower `num_frames` to 33 (~2 s clip) for faster turnaround.

---

## Project structure

```
llm_experiments/
├── llm_chat_app.py            # The main FastAPI chat app + inlined HTML frontend
├── serve_llm.slurm            # SLURM submission script for the chat app
├── flux_gen_app.py            # Flux.1 schnell FastAPI microservice (port 8768)
├── serve_flux_gen.slurm       # SLURM submission script for Flux.1 schnell
├── video_gen_app.py           # Wan2.1 1.3B FastAPI microservice (port 8769)
├── serve_video_gen.slurm      # SLURM submission script for Wan2.1 video generation
├── video_lora_app.py          # Wan2.2 T2V 14B + LoRA microservice (port 8770)
├── serve_video_lora.slurm     # SLURM submission script for Wan2.2 LoRA video generation
├── logs/                      # SLURM output/error logs, one pair per job
├── README.md                  # This file
└── (external)                 # Models live outside the project tree:
    /path/to/gemma4/           #   15 GB Gemma 4 model files
    /path/to/whisper/          #   1.5 GB Whisper medium .pt
    $HF_HOME/hub/...           #   SDXL base (~7 GB) + Lightning UNet (~6 GB) + Flux (~24 GB)
```

The three services are completely independent — start, stop, and restart them on their own schedules. They communicate via plain HTTP on the cluster's internal network, not via shared memory or pipes.

The frontend (HTML, CSS, JavaScript) is embedded as a Python string at the top of `llm_chat_app.py`. There are no separate template or static directories. To change the UI, edit the `HTML` constant in that file.

---

## Licence

This repository contains application code only. Gemma 4 is distributed under Google's Gemma Terms of Use. Whisper is MIT-licensed by OpenAI. Marked.js and DOMPurify (loaded from CDN by the frontend) are MIT-licensed.
