"""
Wan2.2 T2V 14B text-to-video microservice with NSFW LoRA support.

Base model : Wan-AI/Wan2.2-T2V-A14B-Diffusers  (~28 GB bfloat16)
LoRA repo  : lkzd7/WAN2.2_LoraSet_NSFW          (T2V subset, HIGH+LOW pairs)

Wan 2.2 uses two denoisers.  Each LoRA comes as a HIGH-noise file (loaded into
the first denoiser: transformer) and a LOW-noise file (second denoiser:
transformer_2, via load_into_transformer_2=True).

Uses sequential CPU offload so the 28 GB model fits in a 20 GB MIG slice.
Trade-off: generation is slower (~25–45 min for 5 s at 30 steps).

Pairs with llm_chat_app.py via the /videolora command on port 8770.

Run via serve_video_lora.slurm or:
    uvicorn video_lora_app:app --port 8770
"""
import asyncio, base64, os, tempfile, threading

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diffusers import WanPipeline
from diffusers.utils import export_to_video

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID  = os.environ.get("VIDEO_LORA_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
LORA_REPO = os.environ.get("VIDEO_LORA_REPO",  "lkzd7/WAN2.2_LoraSet_NSFW")
PORT      = int(os.environ.get("PORT", 8770))
CACHE_DIR = os.environ.get("HF_HOME", "/scratch/users/t07an25/llm_experiments/hf_cache")

# ── Available LoRAs (T2V subset) ────────────────────────────────────────────
# Each entry has a HIGH-noise file (→ transformer) and a LOW-noise file
# (→ transformer_2).  Only T2V LoRAs are listed here; the I2V ones from the
# same repo require a WanImageToVideoPipeline and are not supported.
LORAS: dict[str, dict] = {
    "doggy": {
        "high": "mql_doggy_a_wan22_t2v_v1_high_noise.safetensors",
        "low":  "mql_doggy_a_wan22_t2v_v1_low_noise.safetensors",
    },
    "spoon": {
        "high": "mqlspn_a_wan22_t2v_v1_high_noise.safetensors",
        "low":  "mqlspn_a_wan22_t2v_v1_low_noise.safetensors",
    },
    "sfbehind": {
        "high": "sfbehind_v2.1_high_noise.safetensors",
        "low":  "sfbehind_v2.1_low_noise.safetensors",
    },
    "transition": {
        "high": "sid3l3g_transition_v2.0_H.safetensors",
        "low":  "sid3l3g_transition_v2.0_L.safetensors",
    },
}

# ── App + state ──────────────────────────────────────────────────────────────
app = FastAPI(title="Wan2.2 Video LoRA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipe: WanPipeline | None = None
_loaded_lora: str | None = None   # name of the currently active LoRA, or None
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
    # 14B model ≈ 28 GB bfloat16.  Sequential offload streams transformer
    # blocks to GPU one at a time — peak VRAM stays under ~12 GB on a 20 GB
    # MIG slice.  Inference is slower as a result.
    pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    print("[startup] Wan2.2 T2V 14B ready.", flush=True)


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)


# ── Request schema ─────────────────────────────────────────────────────────────
class GenRequest(BaseModel):
    prompt: str
    lora: str | None = None        # key from LORAS dict, or None for bare Wan2.2
    lora_scale: float = 1.0
    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "deformed, ugly, watermark, text, low resolution, static, no motion"
    )
    width: int  = 832
    height: int = 480
    num_frames: int = 81            # 5 s @ 16 fps
    num_inference_steps: int = 30   # fewer steps — 14B with seq. offload is slow
    guidance_scale: float = 5.0
    seed: int | None = None


# ── LoRA helpers ───────────────────────────────────────────────────────────────
def _apply_lora(name: str, scale: float) -> None:
    """Load the HIGH+LOW noise LoRA pair for `name`.

    Unloads any previously active LoRA first.  Wan 2.2 uses two denoisers, so
    the HIGH file goes into transformer and the LOW file into transformer_2.
    Skips reload if the same LoRA is already loaded.
    """
    global _loaded_lora
    if _loaded_lora == name:
        print(f"[lora] '{name}' already loaded, reusing.", flush=True)
        return

    if _loaded_lora is not None:
        print(f"[lora] Unloading '{_loaded_lora}'…", flush=True)
        _pipe.unload_lora_weights()
        _loaded_lora = None

    cfg = LORAS[name]
    print(f"[lora] Loading '{name}' (scale={scale})…", flush=True)

    # First denoiser (transformer)
    _pipe.load_lora_weights(
        LORA_REPO,
        weight_name=cfg["high"],
        adapter_name=f"{name}_high",
        cache_dir=CACHE_DIR,
    )
    # Second denoiser (transformer_2)
    _pipe.load_lora_weights(
        LORA_REPO,
        weight_name=cfg["low"],
        adapter_name=f"{name}_low",
        load_into_transformer_2=True,
        cache_dir=CACHE_DIR,
    )
    _pipe.set_adapters(
        [f"{name}_high", f"{name}_low"],
        adapter_weights=[scale, scale],
    )
    _loaded_lora = name
    print(f"[lora] '{name}' active.", flush=True)


def _clear_lora() -> None:
    global _loaded_lora
    if _loaded_lora is not None:
        print(f"[lora] Clearing '{_loaded_lora}'…", flush=True)
        _pipe.unload_lora_weights()
        _loaded_lora = None


# ── API ────────────────────────────────────────────────────────────────────────
@app.get("/ready")
def ready():
    return {"status": "ready" if _pipe else "loading"}


@app.get("/loras")
def list_loras():
    """Return the list of available LoRA names."""
    return list(LORAS.keys())


@app.post("/generate")
async def generate(req: GenRequest):
    """Generate a ~5-second video, optionally with a LoRA applied.

    Returns JSON:
        {"video": "<base64 MP4>", "num_frames": N, "fps": 16,
         "prompt": "...", "lora": "<name>|null"}

    On OOM the service retries at 33 frames (~2 s) and adds a "note" field.
    """
    if not _pipe:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)
    if req.lora and req.lora not in LORAS:
        return JSONResponse(
            {"error": f"Unknown LoRA '{req.lora}'. Available: {list(LORAS.keys())}"},
            status_code=400,
        )

    loop = asyncio.get_event_loop()

    def _run(n_frames: int, is_retry: bool = False):
        with _lock:
            try:
                if req.lora:
                    _apply_lora(req.lora, req.lora_scale)
                else:
                    _clear_lora()

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
                frames = output.frames[0]

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
                    "lora": req.lora,
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

    # OOM retry: 33 frames (~2 s)
    if result.get("oom"):
        print(f"[generate] OOM at {req.num_frames} frames; retrying with 33 …", flush=True)
        result = await loop.run_in_executor(None, lambda: _run(33, is_retry=True))

    if "error" in result:
        return JSONResponse(result, status_code=500)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
