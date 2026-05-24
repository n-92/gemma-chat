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
10. [Configuration reference](#configuration-reference)
11. [API endpoints](#api-endpoints)
12. [How long audio is handled](#how-long-audio-is-handled)
13. [Troubleshooting](#troubleshooting)
14. [Project structure](#project-structure)

---

## Features

- **Multimodal chat** — send text, images, audio, or video files in any combination.
- **Streaming responses** — Gemma 4's tokens arrive in the browser as they are generated (Server-Sent Events).
- **Vision input** — drop any image (JPG/PNG/WebP/etc.) into the chat; Gemma 4 sees it directly.
- **Audio transcription** — any audio file (MP3, WAV, M4A, OGG, FLAC, AAC) is transcribed on the GPU with Whisper medium.
- **Video files** — video files (MP4, WebM, MOV) are accepted; the audio track is extracted by ffmpeg and transcribed. There is no visual frame analysis of videos (see *What it does NOT do*).
- **Chunked summarisation of long audio** — transcripts longer than 1 200 words are automatically split into ~900-word segments, each summarised individually, then a final answer is composed from the summaries. Lets you analyse 30+ minute podcasts on a small GPU.
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

## Configuration reference

All configuration is via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `/scratch/users/t07an25/llm_experiments/gemma4` | Directory holding the Gemma 4 model files |
| `WHISPER_PATH` | `/scratch/users/t07an25/llm_experiments/whisper` | Directory holding the Whisper `.pt` file |
| `PORT` | `8766` | HTTP port for the FastAPI app |

Tunable constants live near the top of `llm_chat_app.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `CHUNK_WORDS` | `900` | Words per chunk in chunked summarisation mode |
| `LONG_TRANSCRIPT_WORDS` | `1200` | Threshold above which a transcript is chunked |

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
| `{"status": "..."}` | Live progress update for the typing bubble |
| `{"transcript": "..."}` | Whisper's output, shown to the user as a separate bubble |
| `{"text": "..."}` | A generation chunk to append to the assistant's response |
| `{"error": "..."}` | Something went wrong; the UI shows it as an error bubble |
| `{"done": true}` | End of stream |

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

---

## Project structure

```
llm_experiments/
├── llm_chat_app.py     # The FastAPI app + inlined HTML frontend
├── serve_llm.slurm     # SLURM submission script (HPC only)
├── logs/               # SLURM output/error logs, one pair per job
├── README.md           # This file
└── (external)          # Models live outside the project tree:
    /path/to/gemma4/    #   15 GB Gemma 4 model files
    /path/to/whisper/   #   1.5 GB Whisper medium .pt
```

The frontend (HTML, CSS, JavaScript) is embedded as a Python string at the top of `llm_chat_app.py`. There are no separate template or static directories. To change the UI, edit the `HTML` constant in that file.

---

## Licence

This repository contains application code only. Gemma 4 is distributed under Google's Gemma Terms of Use. Whisper is MIT-licensed by OpenAI. Marked.js and DOMPurify (loaded from CDN by the frontend) are MIT-licensed.
