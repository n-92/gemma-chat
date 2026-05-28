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
12. [Talking head: Ditto + Chatterbox](#talking-head-ditto--chatterbox)
13. [Visual storytelling: /storyboard and /story](#visual-storytelling-storyboard-and-story)
14. [Configuration reference](#configuration-reference)
15. [API endpoints](#api-endpoints)
16. [How long audio is handled](#how-long-audio-is-handled)
17. [Troubleshooting](#troubleshooting)
18. [Project structure](#project-structure)

---

## Features

- **Multimodal chat** — send text, images, audio, or video files in any combination.
- **Streaming responses** — Gemma 4's tokens arrive in the browser as they are generated (Server-Sent Events).
- **Vision input** — drop any image (JPG/PNG/WebP/etc.) into the chat; Gemma 4 sees it directly.
- **Audio transcription** — any audio file (MP3, WAV, M4A, OGG, FLAC, AAC) is transcribed on the GPU with Whisper medium.
- **Video files** — video files (MP4, WebM, MOV) are accepted; the audio track is extracted by ffmpeg and transcribed. There is no visual frame analysis of videos (see *What it does NOT do*).
- **Chunked summarisation of long audio** — transcripts longer than 1 200 words are automatically split into ~900-word segments, each summarised individually, then a final answer is composed from the summaries. Lets you analyse 30+ minute podcasts on a small GPU.
- **Image generation** — `/imageflux <prompt>` generates a high-quality image (~85 sec) using **Flux.1 schnell**, the only open model that reliably renders readable text in images.
- **Video generation** — `/video <prompt>` — 5-second clip using **Wan2.1 1.3B** (~8 min on a 20 GB MIG slice). Real CFG guidance means the model follows the prompt.
- **Talking head video** — `/talk <text>` — upload a face photo (optional) and any voice clip (optional) then type what you want it to say. **Chatterbox TTS** synthesises the speech (with voice cloning if a clip is attached), **Ditto** animates the face in sync. (~3–5 min on a 20 GB MIG slice).
- **Dual attachments per message** — attach a picture *and* an audio clip at the same time. With `/talk`, the picture is the face and the audio is the voice reference for cloning.
- **Visual storytelling** — `/story <url|text>` turns an article into a narrated visual story: Gemma writes a scene-by-scene storyboard, Chatterbox voices each scene, Flux paints each scene, and ffmpeg adds a Ken-Burns motion pass and stitches the final MP4. `/storyboard <url|text>` previews just the scene list first. **Live, granular progress** — every stage and every scene streams a progress event (with per-scene thumbnails) so you watch the bubble fill in instead of staring at a spinner for ten minutes.
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

| Service | Min VRAM | Disk (weight cache) |
|---|---|---|
| Flux.1 schnell | ~8 GB (with sequential offload) | ~24 GB |
| Wan2.1 1.3B (video) | ~8 GB | ~9 GB |
| Ditto + Chatterbox (talking head) | ~8–12 GB | ~5 GB |

Total disk if you run everything: ~60 GB.

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

## Talking head: Ditto + Chatterbox

The chat app can animate any face photo to say any text in any voice using a two-stage pipeline:

```
                            ┌───────────────────────┐
                            │   ditto_talk_app.py   │
                            │   FastAPI · port 8770 │
       /talk <text>         │                       │
  ┌────┐  + image (face)    │   1) Chatterbox TTS   │   ┌───────────┐
  │chat│ ─ + audio (voice) ─┼──▶  text + voice_ref  │──▶│  WAV 16k  │
  └────┘                    │     → speech WAV      │   └─────┬─────┘
   port 8766                │                       │         │
   (llm_chat_app.py)        │   2) Ditto SDK        │   ┌─────▼─────┐
        ▲                   │      WAV + face       │──▶│   MP4     │
        │  generated_video  │      → talking head   │   └─────┬─────┘
        └───────────────────┤                       │         │
                            │   3) base64 + JSON ◀──┼─────────┘
                            └───────────────────────┘
```

1. **Chatterbox TTS** (Resemble AI) synthesises natural speech from text. With a 5–20 s reference WAV/MP3, it **clones that voice**.
2. **Ditto** (Antgroup) animates the face photo in sync with the audio — lip movement, head pose, blinking. PyTorch backend (no TensorRT compile step).
3. The MP4 is base64-encoded and streamed back to the chat as an SSE event; the browser embeds it inline.

| Property | Value |
|---|---|
| TTS model | `resemble-ai/chatterbox` |
| Video model | `antgroup/ditto-talkinghead` (PyTorch backend, ~5 GB checkpoints) |
| Face input | Single PNG/JPG. Per-request via chat upload, or fallback `TALK_FACE_PATH` |
| Voice cloning | 5–20 s reference WAV/MP3/MP4. Per-request via chat audio upload, or fallback `TALK_VOICE_PATH` |
| Output | MP4 (duration matches the spoken text), played inline |
| Generation time | ~1.5–2 min per 10 s of speech on a 20 GB MIG slice |
| Practical prompt cap | ~80 words (~500 chars) — Chatterbox `max_new_tokens=1000` ≈ 40 s of audio |
| Port | 8770 |

### Step 1 — Clone Ditto and create the conda env

Ditto uses a separate `ditto` env (Python 3.10) from the main `rag_gemma4` chat env. The two ship side-by-side under `~/sharedscratch/.conda/envs/`.

```bash
# On the HPC head node:
git clone https://github.com/antgroup/ditto-talkinghead \
    ~/llm_experiments/ditto-talkinghead
cd ~/llm_experiments/ditto-talkinghead
conda env create -f environment.yaml    # creates env "ditto" with Python 3.10
conda activate ditto
pip install chatterbox-tts              # add TTS on top of the Ditto env
```

> **CentOS 7 / glibc 2.17 caveat.** Ditto's stock `environment.yaml` pins `numpy=2.0.1` and assumes `onnxruntime-gpu>=1.18` — neither works on macleod1's glibc 2.17. After the conda env is created, install the following corrective dep set with `pip --no-deps` so torch/torchaudio versions stay locked:
>
> ```bash
> # numpy back to 1.x — onnxruntime-gpu 1.16.3 is built against numpy 1.x
> pip install --no-deps numpy==1.26.4
>
> # Ditto-side Python deps that environment.yaml leaves out for the PyTorch backend
> pip install --no-deps \
>   filetype==1.2.0 imageio==2.36.1 imageio-ffmpeg==0.5.1 \
>   opencv-python-headless==4.10.0.84 scikit-image==0.25.0 scikit-learn==1.6.0 \
>   tifffile==2024.12.12 numba==0.60.0 llvmlite==0.43.0 audioread==3.0.1 \
>   cython==3.0.11 msgpack==1.1.0 cuda-python==12.6.2.post1 pooch==1.8.2 \
>   joblib==1.4.2 lazy-loader==0.4 threadpoolctl==3.5.0 decorator==5.1.1 \
>   platformdirs==4.3.6 polygraphy colored
>
> # GPU inference for Ditto's auxiliary models (face detect, landmarks)
> # 1.16.3 is the last cp310 wheel that runs on glibc 2.17 (later ones need 2.28)
> pip install --no-deps onnxruntime-gpu==1.16.3
>
> # mediapipe needs protobuf<5; onnx needs protobuf 4.x compatible interface
> pip install --no-deps mediapipe==0.10.14 'protobuf>=4.21,<5'
> pip install --no-deps onnx==1.16.2
>
> # matplotlib (mediapipe drawing utils import it) + pyparsing/cycler/etc.
> pip install --no-deps matplotlib pyparsing cycler kiwisolver fonttools \
>   contourpy python-dateutil attrs flatbuffers absl-py
> ```
>
> The runtime also needs **GCC 14.2 libstdc++ (`CXXABI_1.3.15`)** for `soxr` and **bundled libsndfile 1.0.31** with its full codec chain (FLAC 8, vorbis, opus, ogg). The supplied `serve_ditto_talk.slurm` adds the GCC 14 lib64 path to `LD_LIBRARY_PATH` automatically. The codec libs are copied from `~/sharedscratch/.conda/pkgs/{libsndfile,libflac,libvorbis,libopus,libogg}*/lib/` into the env's `lib/` once during setup.

### Step 2 — Download Ditto checkpoints

```bash
cd ~/llm_experiments/ditto-talkinghead
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

This pulls ~5 GB into `checkpoints/`. Chatterbox downloads automatically from HF on first run (~2 GB).

### Step 3 — Copy your face photo to the HPC

```bash
# From your local machine:
scp -i ~/.ssh/macleod1_key face.png \
    t07an25@macleod1.abdn.ac.uk:~/llm_experiments/face.png
```

Any clear front-facing photo works. The service falls back to `TALK_FACE_PATH` if no image is uploaded in the chat.

### Step 3b — Voice cloning (optional)

Chatterbox can clone any voice from a short clean speech clip. Two ways to wire this up:

**Fixed server-side default** — every `/talk` uses this voice unless overridden:

```bash
# Convert a longer clip to a clean 12 s 24 kHz mono WAV
conda activate ditto
ffmpeg -y -ss 5 -t 12 -i ~/voice_source.mp3 \
       -ac 1 -ar 24000 -c:a pcm_s16le \
       ~/llm_experiments/voice_ref.wav

# In serve_ditto_talk.slurm uncomment:
#   export TALK_VOICE_PATH=/home/$USER/llm_experiments/voice_ref.wav
# Then resubmit the job.
```

**Per-message override** — attach a 5–20 s audio clip in the chat alongside `/talk <text>`. The chat backend sends the bytes as `voice_ref` in the JSON request, the Ditto service writes it to a temp WAV and passes it to Chatterbox's `audio_prompt_path`. Overrides `TALK_VOICE_PATH` for that single message.

### Step 4 — Copy the service files and submit the job

```bash
# From your local machine:
scp -i ~/.ssh/macleod1_key ditto_talk_app.py \
    t07an25@macleod1.abdn.ac.uk:~/llm_experiments/

scp -i ~/.ssh/macleod1_key serve_ditto_talk.slurm \
    t07an25@macleod1.abdn.ac.uk:~/llm_experiments/
```

Then on the HPC:

```bash
cd ~/llm_experiments
sbatch serve_ditto_talk.slurm
tail -f logs/<JOBID>_ditto_talk.out
```

When you see:

```
[startup] Chatterbox ready (sr=24000 Hz).
[startup] Ditto ready.
```

the service is accepting requests.

### Step 5 — Tell the chat app where the service lives

| Variable | Default | Purpose |
|---|---|---|
| `TALK_GEN_URL` | `http://gpu02:8770` | URL of the Ditto talking head service |

If the job lands on a different node, add to `serve_llm.slurm`:

```bash
export TALK_GEN_URL=http://<actual_node>:8770
```

### Step 6 — Use it from the chat UI

You can attach **a face photo and/or a voice clip** in the same message. The upload button accepts both, side-by-side, and the chat backend routes them to the right `/talk` fields.

| Attached | Used for |
|---|---|
| nothing | Server default `TALK_FACE_PATH` and `TALK_VOICE_PATH` |
| image only | Your face + server default voice |
| audio only | Server default face + your voice (clone) |
| image + audio | Your face + your voice |

Examples:

```
/talk Hello world, this is a talking head video.
/talk Welcome to my channel — today we're testing voice cloning.
```

You'll see **🎬 Generating with Ditto: …** in the status bubble, then the video inline.

### Step 7 — Test the service directly (optional)

```bash
curl http://gpu02:8770/ready
# → {"status":"ready"}

# Generate with the server-side face + default voice:
curl -X POST http://gpu02:8770/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, I am a talking head powered by Ditto and Chatterbox."}' \
  --max-time 400 \
  | python -c "
import sys, json, base64
d = json.load(sys.stdin)
open('talk_test.mp4', 'wb').write(base64.b64decode(d['video']))
print('Saved talk_test.mp4')
"

# Generate with a custom face + voice (both base64 in the JSON body):
python - <<'PY'
import base64, json, requests
face  = base64.b64encode(open('face.png','rb').read()).decode()
voice = base64.b64encode(open('voice_ref.wav','rb').read()).decode()
r = requests.post('http://gpu02:8770/generate',
    json={'prompt':'My face, my voice.', 'face_image':face, 'voice_ref':voice},
    timeout=400)
open('talk_custom.mp4','wb').write(base64.b64decode(r.json()['video']))
PY
```

### Step 8 — Stopping the job

```bash
for j in $(squeue -u $USER -h -n ditto_talk -o %i); do scancel $j; done
```

### Quick reference

| Task | Command |
|---|---|
| Clone Ditto repo | `git clone https://github.com/antgroup/ditto-talkinghead ~/llm_experiments/ditto-talkinghead` |
| Create env | `conda env create -f environment.yaml && conda activate ditto && pip install chatterbox-tts` |
| Download checkpoints | `git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints` |
| Copy face photo | `scp face.png t07an25@macleod1.abdn.ac.uk:~/llm_experiments/face.png` |
| Trim a voice ref | `ffmpeg -ss 5 -t 12 -i src.mp3 -ac 1 -ar 24000 voice_ref.wav` |
| Start service | `sbatch serve_ditto_talk.slurm` |
| Check readiness | `curl http://gpu02:8770/ready` |
| Use from chat | Upload face + voice → `/talk <text>` |
| Stop the job | `scancel <JOBID>` |

---

## Visual storytelling: /storyboard and /story

`/story <url|text>` turns an article into a narrated visual story. It is an **orchestrator** — it owns no GPU and loads no model. Instead it fans out over HTTP to the three services you already run, then muxes the result with ffmpeg:

```
                              ┌────────────────────────────────┐
                              │          story_app.py          │
                              │      FastAPI · port 8772       │
   /story <url|text>          │  (CPU-only orchestrator)       │
  ┌────┐                      │                                │      ┌──────────────┐
  │chat│ ── url or text ──────┼─▶ 1) fetch + clean article     │      │  Gemma 4     │
  └────┘                      │   2) storyboard ───────────────┼─────▶│  :8766       │
   port 8766                  │      (N scenes of JSON)         │◀─────│ /generate_text│
   (llm_chat_app.py)          │   3) voiceover per scene ───────┼─────▶│  Chatterbox  │
        ▲                     │                                 │◀─────│ :8770 /tts   │
        │ progress events     │   4) image per scene ───────────┼─────▶│  Flux.1      │
        │ (stage + thumbs)    │                                 │◀─────│ :8768 /generate│
        │                     │   5) ffmpeg Ken-Burns + concat  │      └──────────────┘
        │  generated_video    │   6) base64 MP4 ◀───────────────┤
        └─────────────────────┤      + done                     │
                              └────────────────────────────────┘
```

1. **Fetch** — pulls the URL and strips HTML to text (or you paste the text directly; HPC compute nodes often block outbound HTTP).
2. **Storyboard** — Gemma 4's `/generate_text` returns a strict-JSON array of scenes, each with `narration` (1–2 spoken sentences) and an `image_prompt` (a vivid visual description). By default it **anonymises private individuals** — no real names or faces, people are referred to by role (e.g. "the student").
3. **Voiceover** — each scene's narration goes to Chatterbox `/tts` on the Ditto service (the same default voice as `/talk`; see [Talking head](#talking-head-ditto--chatterbox)). If TTS is unavailable it falls back to a silent cut.
4. **Images** — each `image_prompt` goes to Flux.1 `/generate`. The finished image is also sent to the browser immediately as a **thumbnail** so you see scenes appear one by one.
5. **Render** — ffmpeg applies a slow **Ken-Burns zoom** (`zoompan`) to each still, sets the clip length to the scene's voiceover duration, then concatenates all clips into one MP4 (H.264 + AAC).
6. The MP4 is base64-encoded and streamed back as a `generated_video` SSE event; the browser embeds it inline.

| Property | Value |
|---|---|
| Storyboard model | Gemma 4 27B (reuses the running chat job — no extra GPU) |
| Voice | Chatterbox TTS on the Ditto service (`TTS_URL/tts`) |
| Images | Flux.1 schnell (`FLUX_URL/generate`) |
| Render | ffmpeg `zoompan` Ken-Burns + concat demuxer → H.264/AAC MP4 |
| Default scenes | 8 (override per request with `n_scenes`) |
| Output resolution | 1280×720 @ 30 fps (`STORY_W` / `STORY_H` / `STORY_FPS`) |
| Generation time | ~10–20 min for 8 scenes (script + 8 voiceovers + 8 images + render) |
| GPU | **None** — CPU-only SLURM job; it only calls the other services |
| Port | 8772 |

> **ffmpeg encoder note.** The conda `ditto` env's ffmpeg 4.3 has **no GPL `libx264`**, and its bundled `libopenh264` has a library-version mismatch — neither produces browser-playable H.264. `story_app.py` therefore defaults `FFMPEG_BIN` to the **`imageio-ffmpeg` static binary** already inside the env (`.../imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2`), which ships a working `libx264`. Override with `FFMPEG_BIN` / `STORY_VCODEC` if your build differs.

### Step 1 — Prerequisites

The orchestrator runs in the existing **`ditto`** conda env (it needs only `fastapi`, `uvicorn`, `httpx`, `pydantic`, and an ffmpeg with `libx264` — all already present). The **chat, Flux, and Ditto/Chatterbox services must all be up**, since the orchestrator calls them. The chat app must expose `/generate_text` (added alongside this feature).

### Step 2 — Copy the service files and submit the job

```bash
# From your local machine:
scp -i ~/.ssh/macleod1_key story_app.py \
    t07an25@macleod1.abdn.ac.uk:~/llm_experiments/
scp -i ~/.ssh/macleod1_key serve_story.slurm \
    t07an25@macleod1.abdn.ac.uk:~/llm_experiments/
```

Then on the HPC:

```bash
cd ~/llm_experiments
sbatch serve_story.slurm
curl http://gpu02:8772/ready    # → {"ready": true}
```

`serve_story.slurm` is a **CPU-only** job (no `--gres`) pinned to `gpu02` so the chat app's hard-coded `http://gpu02:8772` resolves. It exports the upstream URLs (`GEMMA_URL`, `FLUX_URL`, `TTS_URL`) and the render settings (`STORY_W`, `STORY_H`, `STORY_FPS`).

### Step 3 — Tell the chat app where the service lives

| Variable | Default | Purpose |
|---|---|---|
| `STORY_GEN_URL` | `http://gpu02:8772` | URL of the story orchestrator |

If the job lands on a different node, add to `serve_llm.slurm`:

```bash
export STORY_GEN_URL=http://<actual_node>:8772
```

### Step 4 — Use it from the chat UI

```
/storyboard https://example.com/some-article     # preview the scene list first
/story      https://example.com/some-article     # full narrated video
/story      Paste the whole article text here ... # if the node can't reach the URL
```

- **`/storyboard`** runs only steps 1–2 and stops at the scene list — fast, so you can sanity-check the narration before committing ~15 minutes to a render.
- **`/story`** runs the whole pipeline. The progress bubble shows a checklist (`fetch → storyboard → voice → image → render`), a live percentage and ETA, and a thumbnail strip that fills in as each scene's image is generated.

If the URL can't be fetched from the compute node (outbound HTTP is often blocked), paste the article text after the command instead.

### Step 5 — Test the service directly (optional)

```bash
curl http://gpu02:8772/ready
# → {"ready": true}

# Storyboard-only (fast), pasting text so no outbound HTTP is needed:
curl -N -X POST http://gpu02:8772/story \
  -H 'Content-Type: application/json' \
  -d '{"text":"<at least ~200 chars of article text>","mode":"storyboard","n_scenes":6}'
# → a stream of `data: {...}` SSE frames ending in {"storyboard": {...}} and {"done": true}
```

### Step 6 — Stopping the job

```bash
for j in $(squeue -u $USER -h -n story_serve -o %i); do scancel $j; done
```

### Quick reference

| Task | Command |
|---|---|
| Copy service files | `scp story_app.py serve_story.slurm t07an25@macleod1.abdn.ac.uk:~/llm_experiments/` |
| Start service | `sbatch serve_story.slurm` |
| Check readiness | `curl http://gpu02:8772/ready` |
| Preview scenes | `/storyboard <url\|text>` |
| Full render | `/story <url\|text>` |
| Stop the job | `scancel <JOBID>` |

> **Requires** the chat (8766), Flux (8768), and Ditto/Chatterbox (8770) services to be running.

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
| `VIDEO_GEN_URL` | `http://gpu02:8769` | chat | Where to find the Wan2.1 1.3B video service |
| `TALK_GEN_URL` | `http://gpu02:8770` | chat | Where to find the Ditto talking head service |
| `STORY_GEN_URL` | `http://gpu02:8772` | chat | Where to find the visual-story orchestrator |
| `GEMMA_URL` | `http://gpu02:8766` | story service | Chat app's `/generate_text` (storyboard) |
| `FLUX_URL` | `http://gpu02:8768` | story service | Flux service `/generate` (per-scene image) |
| `TTS_URL` | `http://gpu02:8770` | story service | Ditto service `/tts` (per-scene voiceover) |
| `STORY_W` / `STORY_H` | `1280` / `720` | story service | Output video resolution |
| `STORY_FPS` | `30` | story service | Output video frame rate |
| `FFMPEG_BIN` | imageio-ffmpeg static binary | story service | ffmpeg with a working `libx264` (auto-detected) |
| `STORY_VCODEC` / `STORY_VBITRATE` | `libx264` / `4M` | story service | Render codec and bitrate |
| `TALK_FACE_PATH` | _(must be set)_ | ditto service | Path to fallback face image on the HPC |
| `TALK_VOICE_PATH` | _(optional)_ | ditto service | Path to ~10 s reference WAV for voice cloning |
| `DITTO_REPO` | `~/llm_experiments/ditto-talkinghead` | ditto service | Path to cloned Ditto repo |
| `DITTO_DATA_ROOT` | `$DITTO_REPO/checkpoints/ditto_pytorch` | ditto service | Ditto PyTorch checkpoint dir |
| `DITTO_CFG` | `$DITTO_REPO/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl` | ditto service | Ditto config pickle |
| `HF_HOME` | `/scratch/users/t07an25/llm_experiments/hf_cache` | image/video/talk services | Where to cache diffusion model weights |
| `HF_TOKEN` | from `~/.huggingface/token` | Flux | HF access token for the gated Flux repo |
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
| `message` | string | User text. Prefix with `/imageflux`, `/video`, `/talk`, `/storyboard`, or `/story` to route to a generation service |
| `history` | string | JSON array of `{role, content}` objects from the previous turns |
| `image` | file | An image. Sent to Gemma 4 for vision, or used as the face for `/talk` |
| `audio` | file | An audio or video file. Whisper-transcribed by default, or used as the voice reference when paired with `/talk` |

**Response:** `text/event-stream`. Each event is a JSON object on a `data:` line:

| Field | Meaning |
|-------|---------|
| `{"status": "..."}` | Live progress update for the typing bubble (transcribing, summarising, generating image, …) |
| `{"transcript": "..."}` | Whisper's output, shown to the user as a separate bubble |
| `{"text": "..."}` | A generation chunk to append to the assistant's response |
| `{"generated_image": "data:image/png;base64,...", "prompt": "...", "model": "..."}` | A generated image to embed in the chat (from `/imageflux`) |
| `{"generated_video": "<base64 MP4>", "prompt": "...", "model": "...", "num_frames": N, "fps": 16}` | A generated video to embed in the chat (from `/video`, `/talk`, or `/story`) |
| `{"progress": {"stage": "...", "label": "...", "step": N, "total": N, "pct": N, "eta_s": N, "thumb": "data:image/png;base64,..."}}` | Live story-pipeline progress (from `/story`); `thumb` present once a scene image is ready |
| `{"storyboard": {"scenes": [...], "n": N}}` | The scene list (from `/storyboard` and `/story`) |
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

#### Talking-head service

`ditto_talk_app.py` exposes:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/ready` | Returns `{"status": "ready"}` once both Chatterbox and Ditto are loaded |
| `POST` | `/generate` | Generates one talking-head video clip |
| `POST` | `/tts` | Text → speech only (no video). Used by the story orchestrator for voiceover |

**POST `/generate` body (JSON):**

```json
{
  "prompt": "Hello, this is a test of the Ditto talking head service.",
  "face_image": "<base64 PNG/JPG>",   // optional — falls back to TALK_FACE_PATH
  "voice_ref":  "<base64 WAV/MP3>",   // optional — falls back to TALK_VOICE_PATH
  "exaggeration": 0.5,                  // 0 = neutral, 1 = highly expressive
  "cfg_weight":   0.5                   // Chatterbox CFG weight
}
```

**Response:**

```json
{
  "video": "<base64-encoded MP4>",
  "prompt": "Hello, this is a test of the Ditto talking head service."
}
```

**POST `/tts` body (JSON):** `{"text": "...", "voice_ref": "<base64 WAV/MP3>"?, "exaggeration": 0.5, "cfg_weight": 0.5}` → returns `{"audio": "<base64 WAV>", "sr": 24000}`. Like `/generate`, an omitted `voice_ref` falls back to `TALK_VOICE_PATH`.

Requests are serialised by a `threading.Lock` — concurrent calls queue rather than fight for VRAM.

#### Visual-story orchestrator

`story_app.py` is a CPU-only orchestrator (no model, no GPU):

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/ready` | Returns `{"ready": true}` |
| `POST` | `/story` | Streams the whole pipeline as SSE (`text/event-stream`) |

**POST `/story` body (JSON):**

```json
{
  "url": "https://example.com/article",   // optional — fetched + stripped to text
  "text": "Paste article text instead",   // optional — used if the node can't reach the URL
  "n_scenes": 8,                            // optional, default 8
  "anonymize": true,                        // optional, default true — no real names/faces
  "mode": "render",                         // "storyboard" = preview only, "render" = full
  "storyboard": null                        // optional — reuse an approved scene list
}
```

**Response:** `text/event-stream` of `data:` frames — `progress`, `storyboard`, `generated_video`, `error`, and `done` events (see the `/chat` SSE table above). The chat app relays these frames straight through to the browser.

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

**`/talk` returns `TypeError: cannot unpack non-iterable NoneType object`**
Ditto's face landmark detector returned `None` — the face in your photo is too small, partially occluded, or at an extreme angle. Use a clear front-facing portrait at least 256×256.

**`/talk` returns `libsndfile.so: cannot open shared object file`**
Compute nodes can't see the OS `libsndfile`. Copy a conda-bundled one into the env lib:
```bash
cp ~/sharedscratch/.conda/pkgs/libsndfile-1.0.31-h9c3ff4c_1/lib/libsndfile.so.1.0.31 \
   ~/sharedscratch/.conda/envs/ditto/lib/
# plus libFLAC.so.8, libvorbis.so.0.4.9, libvorbisenc.so.2.0.12, libopus.so.0, libogg.so.0
# from the matching package dirs under ~/sharedscratch/.conda/pkgs/
```

**`/talk` returns `_ARRAY_API not found` (onnxruntime)**
NumPy 2.x is incompatible with `onnxruntime-gpu 1.16.3` (the latest cp310 wheel that runs on glibc 2.17). Downgrade:
```bash
pip install --no-deps numpy==1.26.4
```

**`/talk` returns `'MessageFactory' object has no attribute 'GetPrototype'`**
Protobuf 5.x conflict — `mediapipe<0.10.18` needs protobuf 4.x while modern `onnx` needs protobuf 5. Pin both:
```bash
pip install --no-deps 'protobuf>=4.21,<5' onnx==1.16.2
```

**`/talk` produces audio but the face barely moves**
Use a higher-quality face photo with the head filling most of the frame. The default Ditto `overall_ctrl_info` has `delta_pitch=2` which is subtle — increase by passing `exaggeration` ≥ 0.7 in the request.

**Voice cloning gives a robotic / unrelated voice**
Reference clip too short, too noisy, or the wrong format. Aim for 8–15 seconds of clean speech, single speaker, no music, encoded as 24 kHz mono WAV (`ffmpeg -ac 1 -ar 24000`).

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

**`/talk` returns "No face image provided and TALK_FACE_PATH not set"**
Either upload a face photo in the chat before sending `/talk`, or SCP a face image to the HPC and set `TALK_FACE_PATH` in `serve_ditto_talk.slurm`.

**`/talk` returns "stream_pipeline_offline not found" or similar ImportError**
The Ditto repo path is wrong. Check that `DITTO_REPO` in `serve_ditto_talk.slurm` points at the cloned `ditto-talkinghead` directory and that `sys.path.insert(0, DITTO_REPO)` at the top of `ditto_talk_app.py` is present.

**`/talk` service is stuck loading / never reaches "Ditto ready"**
Check the job log: `tail logs/<JOBID>_ditto_talk.out`. Most likely cause is the checkpoints directory not existing — run the `git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints` step inside the Ditto repo.

**`/talk` produces audio but the mouth doesn't move**
Ditto requires audio at exactly 16 kHz. The service resamples Chatterbox output automatically, but if you see a Ditto-side error in the log about sample rate, check that `torchaudio` is installed in the `ditto` conda env.

**`/talk` times out (600 s)**
Reduce the text length — longer speech = more video frames = longer Ditto inference. Alternatively raise `timeout` for `kind == "talk"` in `llm_chat_app.py`.

**`export_to_video` fails with `No module named 'imageio'`**
Install the imageio backend: `pip install imageio imageio-ffmpeg`.

**Video output is a corrupted file / browser shows a broken video player**
This can happen if the video job OOMed mid-frame and returned partial data. Check the video job log (`tail logs/<JOBID>_video.out`) for an OOM traceback. The OOM retry (33 frames) should prevent this, but if the retry also OOMed, you'll need a bigger MIG slice or to lower `num_frames` in `GenRequest`.

**Generated video ignores the prompt / output looks like random noise**
Make sure the job is running the Wan2.1 service (`video_gen_app.py`), not an older LTX-Video version. The `guidance_scale=5.0` default in Wan2.1 is what makes the model follow prompts — confirm this in the request by checking the job log.

**`ModuleNotFoundError: No module named 'diffusers'` in video-gen log**
The video job didn't activate the conda env. Edit `serve_video_gen.slurm` and check the `conda activate rag_gemma4` line runs before `uvicorn`.


---

## Project structure

```
llm_experiments/
├── llm_chat_app.py            # Main FastAPI chat app + inlined HTML frontend
├── serve_llm.slurm            # SLURM script for the chat app (port 8766)
├── flux_gen_app.py            # Flux.1 schnell image microservice (port 8768)
├── serve_flux_gen.slurm       # SLURM script for Flux.1 schnell
├── video_gen_app.py           # Wan2.1 1.3B video microservice (port 8769)
├── serve_video_gen.slurm      # SLURM script for Wan2.1 video generation
├── ditto_talk_app.py          # Ditto + Chatterbox talking head microservice (port 8770)
├── serve_ditto_talk.slurm     # SLURM script for Ditto talking head
├── story_app.py               # Visual-story orchestrator (CPU-only, port 8772)
├── serve_story.slurm          # SLURM script for the story orchestrator
├── face.png                   # Default face image for /talk (SCP from local machine)
├── voice_ref.wav              # Default voice clip for /talk cloning (optional, ~10 s)
├── logs/                      # SLURM output/error logs, one pair per job
├── ditto-talkinghead/         # Cloned antgroup/ditto-talkinghead repo
│   └── checkpoints/           #   ~5 GB Ditto model weights
├── README.md                  # This file
└── (external)                 # Models live outside the project tree:
    /path/to/gemma4/           #   15 GB Gemma 4 model files
    /path/to/whisper/          #   1.5 GB Whisper medium .pt
    $HF_HOME/hub/...           #   Flux (~24 GB) + Wan2.1 (~9 GB) + Chatterbox (~2 GB)
```

The services (chat, Flux, Wan2.1 video, Ditto talking head) are completely independent — start, stop, and restart them on their own schedules. They communicate via plain HTTP on the cluster's internal network, not via shared memory or pipes. Each owns its own 20 GB MIG slice on the same A100 (`gpu02` on macleod1). The visual-story orchestrator is the exception: it owns **no GPU** and simply fans out HTTP calls to the chat, Flux, and Ditto services, so it adds a feature without consuming another MIG slice.

The frontend (HTML, CSS, JavaScript) is embedded as a Python string at the top of `llm_chat_app.py`. There are no separate template or static directories. To change the UI, edit the `HTML` constant in that file.

---

## Licence

This repository contains application code only. Gemma 4 is distributed under Google's Gemma Terms of Use. Whisper is MIT-licensed by OpenAI. Marked.js and DOMPurify (loaded from CDN by the frontend) are MIT-licensed.
