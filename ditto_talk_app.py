"""
Ditto Talking Head microservice.

Pipeline: text → Chatterbox TTS → WAV (16 kHz) → Ditto → talking head MP4

Endpoints:
  GET  /ready     → {"status": "ready"|"loading"}
  POST /generate  → {"video": "<base64-mp4>", "prompt": "..."}

Request body (JSON):
  prompt       str   — text to speak
  face_image   str?  — base64-encoded PNG/JPG; falls back to TALK_FACE_PATH env var
  exaggeration float — Chatterbox emotion 0 (neutral) → 1 (expressive), default 0.5
  cfg_weight   float — Chatterbox CFG weight, default 0.5

Environment variables:
  DITTO_REPO       path to cloned antgroup/ditto-talkinghead repo
  DITTO_DATA_ROOT  path to PyTorch checkpoint dir inside the repo
  DITTO_CFG        path to cfg .pkl file (PyTorch backend)
  TALK_FACE_PATH   fallback face image (required when no face_image in request)
  TALK_VOICE_PATH  optional ~10 s reference WAV for Chatterbox voice cloning
  PORT             default 8770
  HF_HOME          HuggingFace cache dir

Setup (run on HPC before submitting the SLURM job):
  # 1. Clone Ditto and create its conda env
  git clone https://github.com/antgroup/ditto-talkinghead ~/llm_experiments/ditto-talkinghead
  cd ~/llm_experiments/ditto-talkinghead
  conda env create -f environment.yaml          # creates env named "ditto" (Python 3.10)
  conda activate ditto
  pip install chatterbox-tts                    # add TTS on top

  # 2. Download model checkpoints (requires git-lfs)
  git lfs install
  git clone https://huggingface.co/digital-avatar/ditto-talkinghead \
      ~/llm_experiments/ditto-talkinghead/checkpoints

  # 3. SCP a face photo to the HPC
  #    (from your local machine)
  scp -i ~/.ssh/macleod1_key face.png t07an25@macleod1.abdn.ac.uk:~/llm_experiments/face.png

  # 4. Copy this file and submit the SLURM job
  scp -i ~/.ssh/macleod1_key ditto_talk_app.py t07an25@macleod1.abdn.ac.uk:~/llm_experiments/
  sbatch serve_ditto_talk.slurm

Pairs with llm_chat_app.py via /talk command on port 8770.
"""
import asyncio, base64, os, sys, tempfile, threading
from pathlib import Path

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
DITTO_REPO      = os.environ.get("DITTO_REPO",
                    str(Path.home() / "llm_experiments/ditto-talkinghead"))
DITTO_DATA_ROOT = os.environ.get("DITTO_DATA_ROOT",
                    f"{DITTO_REPO}/checkpoints/ditto_pytorch")
DITTO_CFG       = os.environ.get("DITTO_CFG",
                    f"{DITTO_REPO}/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
TALK_FACE_PATH  = os.environ.get("TALK_FACE_PATH", "")
TALK_VOICE_PATH = os.environ.get("TALK_VOICE_PATH", "")
PORT            = int(os.environ.get("PORT", 8770))
HF_CACHE        = os.environ.get("HF_HOME",
                    "/scratch/users/t07an25/llm_experiments/hf_cache")

# Ditto requires audio at 16 kHz
DITTO_SAMPLE_RATE = 16_000

# Ensure ditto-talkinghead is importable (stream_pipeline_offline, inference, …)
sys.path.insert(0, DITTO_REPO)

# ── App + state ───────────────────────────────────────────────────────────────
app = FastAPI(title="Ditto Talking Head")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_tts  = None   # ChatterboxTTS
_sdk  = None   # Ditto StreamSDK
_lock = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_models():
    global _tts, _sdk

    # 1. Chatterbox TTS
    print("[startup] Loading Chatterbox TTS …", flush=True)
    from chatterbox.tts import ChatterboxTTS
    _tts = ChatterboxTTS.from_pretrained(device="cuda")
    print(f"[startup] Chatterbox ready (sr={_tts.sr} Hz).", flush=True)

    # 2. Ditto StreamSDK (PyTorch backend — safer on MIG than TensorRT)
    print(f"[startup] Loading Ditto from {DITTO_DATA_ROOT} …", flush=True)
    from stream_pipeline_offline import StreamSDK
    _sdk = StreamSDK(DITTO_CFG, DITTO_DATA_ROOT)
    print("[startup] Ditto ready.", flush=True)


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models)


# ── Request schema ─────────────────────────────────────────────────────────────
class TalkRequest(BaseModel):
    prompt:       str
    face_image:   str | None = None   # base64 PNG/JPG; falls back to TALK_FACE_PATH
    voice_ref:    str | None = None   # base64 WAV/MP3 (5-20s clean speech); falls back to TALK_VOICE_PATH
    exaggeration: float = 0.5         # 0 = neutral, 1 = highly expressive
    cfg_weight:   float = 0.5         # Chatterbox CFG weight


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/ready")
def ready():
    return {"status": "ready" if (_tts and _sdk) else "loading"}


@app.post("/generate")
async def generate(req: TalkRequest):
    """Generate a talking head video from text + optional face image.

    Returns JSON: {"video": "<base64-mp4>", "prompt": "..."}
    """
    if not _tts or not _sdk:
        return JSONResponse({"error": "Models not loaded yet"}, status_code=503)

    if not req.face_image and not TALK_FACE_PATH:
        return JSONResponse(
            {"error": "No face image provided and TALK_FACE_PATH env var not set"},
            status_code=400,
        )

    loop = asyncio.get_event_loop()

    def _run():
        with _lock:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp = Path(tmp_dir)

                    # ── 1. Resolve face image ──────────────────────────────────
                    if req.face_image:
                        face_path = tmp / "face.png"
                        face_path.write_bytes(base64.b64decode(req.face_image))
                    else:
                        face_path = Path(TALK_FACE_PATH)
                    if not face_path.exists():
                        return {"error": f"Face image not found: {face_path}"}

                    # ── 2. Chatterbox TTS: text → WAV ──────────────────────────
                    # Per-request voice_ref overrides the server-side default.
                    if req.voice_ref:
                        vref_path = tmp / "voice_ref.audio"
                        vref_path.write_bytes(base64.b64decode(req.voice_ref))
                        voice_ref = str(vref_path)
                    else:
                        voice_ref = TALK_VOICE_PATH or None
                    print(f"[generate] voice_ref={voice_ref or '<chatterbox default>'}", flush=True)
                    wav_tensor = _tts.generate(
                        req.prompt,
                        audio_prompt_path=voice_ref,
                        exaggeration=req.exaggeration,
                        cfg_weight=req.cfg_weight,
                    )
                    print(f"[generate] TTS done — {wav_tensor.shape[-1] / _tts.sr:.1f}s audio", flush=True)

                    # Resample to 16 kHz if needed (Ditto hard requirement)
                    if _tts.sr != DITTO_SAMPLE_RATE:
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=_tts.sr, new_freq=DITTO_SAMPLE_RATE
                        )
                        wav_tensor = resampler(wav_tensor)

                    wav_path = str(tmp / "speech.wav")
                    torchaudio.save(wav_path, wav_tensor, DITTO_SAMPLE_RATE)

                    # ── 3. Ditto: WAV + image → MP4 ────────────────────────────
                    from inference import run as ditto_run
                    out_path = str(tmp / "output.mp4")
                    ditto_run(_sdk, wav_path, str(face_path), out_path)
                    print(f"[generate] Ditto done → {out_path}", flush=True)

                    # ── 4. Encode and return ───────────────────────────────────
                    video_bytes = Path(out_path).read_bytes()
                    torch.cuda.empty_cache()
                    return {
                        "video":  base64.b64encode(video_bytes).decode(),
                        "prompt": req.prompt,
                    }

            except Exception as exc:
                import traceback
                print(
                    f"[generate error] {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
                torch.cuda.empty_cache()
                return {"error": f"{type(exc).__name__}: {exc}"}

    result = await loop.run_in_executor(None, _run)
    if "error" in result:
        return JSONResponse(result, status_code=500)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
