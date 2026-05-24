"""
SDXL Lightning image-generation microservice.

Pairs with llm_chat_app.py: a separate SLURM job on its own MIG slice that
generates images on demand. The chat app POSTs to /generate and embeds the
returned PNG inline in the conversation.

Run via serve_image_gen.slurm or:
    uvicorn image_gen_app:app --port 8767
"""
import asyncio, base64, io, os, threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = os.environ.get("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_CKPT = "sdxl_lightning_4step_unet.safetensors"
N_STEPS       = 4   # paired with sdxl_lightning_4step_unet.safetensors
PORT          = int(os.environ.get("PORT", 8767))
CACHE_DIR     = os.environ.get("HF_HOME", "/scratch/users/t07an25/llm_experiments/hf_cache")

# ── App + state ───────────────────────────────────────────────────────────────
app = FastAPI(title="SDXL Lightning")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipe: StableDiffusionXLPipeline | None = None
_lock = threading.Lock()

# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global _pipe
    print(f"[startup] Loading SDXL base from {BASE_MODEL} …", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, variant="fp16",
        cache_dir=CACHE_DIR,
    ).to("cuda")

    print(f"[startup] Patching UNet with SDXL Lightning ({N_STEPS}-step) …", flush=True)
    unet_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT, cache_dir=CACHE_DIR)
    unet = UNet2DConditionModel.from_config(pipe.unet.config).to("cuda", torch.float16)
    unet.load_state_dict(load_file(unet_path, device="cuda"))
    pipe.unet = unet

    # SDXL Lightning uses Euler with trailing timestep spacing
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    print("[startup] SDXL Lightning ready.", flush=True)


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)


# ── API ───────────────────────────────────────────────────────────────────────
class GenRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
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
                gen = torch.Generator(device="cuda").manual_seed(int(req.seed))
            with torch.no_grad():
                img = _pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    num_inference_steps=N_STEPS,
                    guidance_scale=0.0,           # required for Lightning
                    width=req.width, height=req.height,
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
