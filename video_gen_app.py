"""
Wan2.1 1.3B text-to-video microservice.

Uses Wan-AI/Wan2.1-T2V-1.3B-Diffusers — ~3 min per 5-second clip on A100 MIG 20 GB.
Proper CFG support (guidance_scale > 1) means it actually follows the prompt,
unlike LTX-Video distilled which is locked at guidance_scale=1.0.

Pairs with llm_chat_app.py via /generate endpoint on port 8769.

Run via serve_video_gen.slurm or:
    uvicorn video_gen_app:app --port 8769
"""
import asyncio, base64, io, os, tempfile, threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diffusers import WanPipeline
from diffusers.utils import export_to_video

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID  = os.environ.get("VIDEO_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
PORT      = int(os.environ.get("PORT", 8769))
CACHE_DIR = os.environ.get("HF_HOME", "/scratch/users/t07an25/llm_experiments/hf_cache")

# ── App + state ───────────────────────────────────────────────────────────────
app = FastAPI(title="Wan2.1 Video")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipe: WanPipeline | None = None
_lock = threading.Lock()

# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global _pipe
    print(f"[startup] Loading {MODEL_ID} …", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )
    # Model is ~8 GB total — fits comfortably in 20 GB MIG.
    # Use model CPU offload (moves sub-models to CPU between uses) for safety.
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    print("[startup] Wan2.1 1.3B ready.", flush=True)


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)


# ── Request schema ────────────────────────────────────────────────────────────
class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "deformed, ugly, watermark, text, low resolution, static, no motion"
    )
    width: int  = 832   # 480P widescreen — recommended for 1.3B
    height: int = 480
    num_frames: int = 81          # 5 s @ 16 fps
    num_inference_steps: int = 50  # standard (non-distilled) — proper CFG
    guidance_scale: float = 5.0   # real CFG — key advantage over LTX distilled
    seed: int | None = None


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/ready")
def ready():
    return {"status": "ready" if _pipe else "loading"}


@app.post("/generate")
async def generate(req: GenRequest):
    """Generate a ~5-second video from a text prompt.

    Returns JSON: {"video": "<base64-encoded MP4>", "num_frames": N, "fps": 16}
    """
    if not _pipe:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    loop = asyncio.get_event_loop()

    def _run(n_frames: int, is_retry: bool = False):
        with _lock:
            try:
                generator = None
                if req.seed is not None:
                    generator = torch.Generator(device="cpu").manual_seed(req.seed)

                output = _pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    width=req.width,
                    height=req.height,
                    num_frames=n_frames,
                    num_inference_steps=req.num_inference_steps,
                    guidance_scale=req.guidance_scale,
                    generator=generator,
                )
                frames = output.frames[0]  # list of PIL images

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    tmp_path = f.name
                export_to_video(frames, tmp_path, fps=16)
                with open(tmp_path, "rb") as f:
                    video_bytes = f.read()
                os.unlink(tmp_path)

                torch.cuda.empty_cache()
                result = {
                    "video": base64.b64encode(video_bytes).decode(),
                    "num_frames": len(frames),
                    "fps": 16,
                    "prompt": req.prompt,
                }
                if is_retry:
                    result["note"] = "OOM on first attempt; fell back to 33 frames."
                return result

            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                return {"oom": True, "error": str(e)}
            except Exception as e:
                import traceback
                print(f"[generate error] {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
                torch.cuda.empty_cache()
                return {"error": f"{type(e).__name__}: {e}"}

    # First attempt
    result = await loop.run_in_executor(None, lambda: _run(req.num_frames))

    # OOM retry: 33 frames (~2s)
    if result.get("oom"):
        print(f"[generate] OOM at {req.num_frames} frames; retrying with 33 …", flush=True)
        result = await loop.run_in_executor(None, lambda: _run(33, is_retry=True))

    if "error" in result:
        return JSONResponse(result, status_code=500)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
