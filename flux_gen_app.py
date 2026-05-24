"""
Flux.1 schnell image-generation microservice.

A second image-gen backend alongside SDXL Lightning. Flux is a generation
ahead on text rendering, hands and prompt fidelity, at the cost of being
larger (~24 GB FP16). To fit on a 20 GB MIG slice we run in bfloat16 with
diffusers' model_cpu_offload, which streams components between CPU↔GPU
as needed — a bit slower than SDXL Lightning but well worth it.

Run via serve_flux_gen.slurm or:
    uvicorn flux_gen_app:app --port 8768
"""
import asyncio, base64, io, os, threading

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diffusers import FluxPipeline

# ── Config ────────────────────────────────────────────────────────────────────
FLUX_MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
N_STEPS    = 4   # Flux schnell is a 1–4 step distillation
PORT       = int(os.environ.get("PORT", 8768))
CACHE_DIR  = os.environ.get("HF_HOME", "/scratch/users/t07an25/llm_experiments/hf_cache")

# ── App + state ───────────────────────────────────────────────────────────────
app = FastAPI(title="Flux.1 schnell")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipe: FluxPipeline | None = None
_lock = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global _pipe
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[startup] Loading Flux.1 schnell from {FLUX_MODEL} …", flush=True)

    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )

    # Flux transformer alone is ~24 GB in bf16 — too big for a 20 GB MIG slice.
    # Sequential CPU offload streams individual layers between CPU and GPU,
    # keeping peak VRAM under ~8 GB. ~30-60 sec per image but no OOM.
    pipe.enable_sequential_cpu_offload()
    # VAE tiling cuts the decode-time VRAM spike on 1024×1024.
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    _pipe = pipe
    print("[startup] Flux.1 schnell ready.", flush=True)


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)


# ── API ───────────────────────────────────────────────────────────────────────
class GenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int | None = None


@app.get("/ready")
def ready():
    return {"ready": _pipe is not None}


@app.post("/generate")
async def generate(req: GenRequest):
    if _pipe is None:
        return JSONResponse({"error": "Model not loaded yet"}, status_code=503)

    loop = asyncio.get_event_loop()

    def run():
        with _lock:
            gen = None
            if req.seed is not None:
                gen = torch.Generator(device="cpu").manual_seed(int(req.seed))
            with torch.no_grad():
                img = _pipe(
                    prompt=req.prompt,
                    num_inference_steps=N_STEPS,
                    guidance_scale=0.0,           # schnell needs CFG=0
                    width=req.width, height=req.height,
                    max_sequence_length=256,
                    generator=gen,
                ).images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()

    try:
        png_bytes = await loop.run_in_executor(None, run)
    except torch.cuda.OutOfMemoryError as ex:
        torch.cuda.empty_cache()
        return JSONResponse({"error": f"GPU OOM: {ex}"}, status_code=503)
    except Exception as ex:
        return JSONResponse({"error": str(ex)}, status_code=500)

    b64 = base64.b64encode(png_bytes).decode()
    return {"image": f"data:image/png;base64,{b64}", "prompt": req.prompt}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
