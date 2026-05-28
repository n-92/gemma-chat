"""
Visual-story orchestrator service.

Turns an article (URL or pasted text) into a narrated visual story by fanning
out across the services you already run:

    Gemma 4   (chat app, :8766 /generate_text)  → storyboard (scene beats)
    Chatterbox(ditto app, :8770 /tts)           → voiceover per scene
    Flux.1    (flux app,  :8768 /generate)       → one image per scene
    ffmpeg                                        → Ken-Burns motion + mux

The whole job is SLOW (script + N voiceovers + N images + render), so the
design goal here is VISIBLE PROGRESS: every stage and every scene emits a
Server-Sent-Events `progress` event that the chat UI relays straight to the
browser. The bubble fills in live (including per-scene thumbnails) instead of
showing one dead spinner for ten minutes.

Run:  uvicorn story_app:app --host 0.0.0.0 --port 8772
Drive it from the chat UI with /storyboard <url|text> (preview only) or
/story <url|text> (full render).
"""
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Upstream services (override via env in the SLURM script) ──────────────────
GEMMA_URL = os.environ.get("GEMMA_URL", "http://gpu02:8766")   # /generate_text
FLUX_URL  = os.environ.get("FLUX_URL",  "http://gpu02:8768")   # /generate  → image
TTS_URL   = os.environ.get("TTS_URL",   "http://gpu02:8770")   # /tts       → wav (b64)
PORT      = int(os.environ.get("PORT", 8772))

# ── Render settings ───────────────────────────────────────────────────────────
VID_W      = int(os.environ.get("STORY_W", 1280))
VID_H      = int(os.environ.get("STORY_H", 720))
FPS        = int(os.environ.get("STORY_FPS", 30))
NO_AUDIO_SCENE_SECS = 4.0          # scene length when a voiceover is unavailable
DEFAULT_SCENES      = 8
# The conda ffmpeg (4.3) lacks GPL libx264 and its bundled libopenh264 has a
# library-version mismatch, so neither produces browser-playable H.264. The
# imageio-ffmpeg static binary (shipped inside the ditto env) DOES have libx264,
# so default to it. Override with FFMPEG_BIN if it moves.
_IMAGEIO_FFMPEG = ("/home/t07an25/sharedscratch/.conda/envs/ditto/lib/python3.10/"
                   "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2")
FFMPEG = os.environ.get(
    "FFMPEG_BIN",
    _IMAGEIO_FFMPEG if os.path.exists(_IMAGEIO_FFMPEG)
    else (shutil.which("ffmpeg") or "ffmpeg"),
)
VCODEC   = os.environ.get("STORY_VCODEC", "libx264")
VBITRATE = os.environ.get("STORY_VBITRATE", "4M")

app = FastAPI(title="Visual Story Orchestrator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── SSE helpers ────────────────────────────────────────────────────────────────
def sse(payload: dict) -> str:
    """Serialise one Server-Sent-Events frame."""
    return f"data: {json.dumps(payload)}\n\n"


def progress(stage: str, label: str, step: int = 0, total: int = 0,
             pct: int = 0, eta_s: int = 0, thumb: str | None = None) -> str:
    ev = {"progress": {"stage": stage, "label": label, "step": step,
                       "total": total, "pct": pct, "eta_s": eta_s}}
    if thumb:
        ev["progress"]["thumb"] = thumb
    return sse(ev)


# ── Request model ────────────────────────────────────────────────────────────
class StoryRequest(BaseModel):
    url: str | None = None
    text: str | None = None
    n_scenes: int = DEFAULT_SCENES
    anonymize: bool = True
    mode: str = "render"                 # "storyboard" = preview only, "render" = full
    storyboard: list | None = None       # reuse an approved storyboard for render
    style: str | None = None             # global art style appended to every scene image


# ── Stage 1: get the article text ───────────────────────────────────────────
async def fetch_article(url: str) -> str:
    """Best-effort fetch + crude HTML→text. HPC compute nodes often block
    outbound HTTP — if this fails the caller should paste the text instead."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StoryBot/1.0)"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as c:
        r = await c.get(url)
        r.raise_for_status()
    html = r.text
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|h[1-6]|li)>", "\n", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


# ── Stage 2: storyboard via Gemma ────────────────────────────────────────────
STORYBOARD_SYS = (
    "You are a video story editor. You turn an article into a tight, vivid "
    "narrated storyboard for a short documentary-style video. Output STRICT JSON "
    "only — no prose, no markdown fences."
)


def storyboard_prompt(article: str, n_scenes: int, anonymize: bool) -> str:
    privacy = (
        "IMPORTANT: anonymise any private individual — never use a real person's "
        "name or describe their face/photo. Refer to them by role (e.g. 'the student'). "
        if anonymize else ""
    )
    return (
        f"Article:\n\"\"\"\n{article[:8000]}\n\"\"\"\n\n"
        f"Write a {n_scenes}-scene narrated storyboard. {privacy}"
        "Keep the drama and tension of the original. Return a JSON array; each element:\n"
        '  {"n": <int>, "narration": "<1-2 punchy spoken sentences>", '
        '"image_prompt": "<vivid VISUAL description for an image generator, '
        'documentary/investigation-board style, no real faces or logos>"}\n'
        "Return ONLY the JSON array."
    )


def _extract_json_array(s: str) -> list:
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model output")
    return json.loads(s[start:end + 1])


async def make_storyboard(article: str, n_scenes: int, anonymize: bool) -> list:
    payload = {
        "system": STORYBOARD_SYS,
        "prompt": storyboard_prompt(article, n_scenes, anonymize),
        "max_new_tokens": 1400,
    }
    async with httpx.AsyncClient(timeout=300.0) as c:
        r = await c.post(f"{GEMMA_URL}/generate_text", json=payload)
        r.raise_for_status()
        raw = r.json().get("text", "")
    scenes = _extract_json_array(raw)
    # Normalise / clamp
    clean = []
    for i, sc in enumerate(scenes[:n_scenes], 1):
        clean.append({
            "n": i,
            "narration": str(sc.get("narration", "")).strip(),
            "image_prompt": str(sc.get("image_prompt", "")).strip(),
        })
    if not clean:
        raise ValueError("storyboard came back empty")
    return clean


# ── Stage 3 helpers: TTS + image ─────────────────────────────────────────────
async def tts_scene(text: str) -> bytes | None:
    """Returns WAV bytes, or None if the TTS service has no /tts endpoint yet."""
    try:
        async with httpx.AsyncClient(timeout=180.0) as c:
            r = await c.post(f"{TTS_URL}/tts", json={"text": text})
        if r.status_code != 200:
            return None
        b64 = r.json().get("audio")
        return base64.b64decode(b64) if b64 else None
    except Exception:
        return None


async def flux_scene(prompt: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=300.0) as c:
            r = await c.post(f"{FLUX_URL}/generate", json={"prompt": prompt})
        if r.status_code != 200:
            return None
        img = r.json().get("image", "")
        if img.startswith("data:"):
            img = img.split(",", 1)[1]
        return base64.b64decode(img) if img else None
    except Exception:
        return None


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return NO_AUDIO_SCENE_SECS


# ── Stage 4: ffmpeg render (Ken-Burns per scene, then concat) ────────────────
def render_scene_clip(img_path: Path, wav_path: Path | None,
                      out_path: Path, dur: float) -> None:
    """One still → a slowly zooming clip of length `dur`, with audio if present."""
    nframes = max(1, int(dur * FPS))
    vf = (
        f"scale={VID_W}:{VID_H}:force_original_aspect_ratio=increase,"
        f"crop={VID_W}:{VID_H},"
        f"zoompan=z='min(zoom+0.0006,1.15)':d={nframes}:s={VID_W}x{VID_H}:fps={FPS}"
    )
    cmd = [FFMPEG, "-y", "-loop", "1", "-i", str(img_path)]
    if wav_path is not None:
        cmd += ["-i", str(wav_path)]
    cmd += ["-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]"]
    if wav_path is not None:
        cmd += ["-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-t", f"{dur:.3f}", "-c:v", VCODEC, "-b:v", VBITRATE,
            "-pix_fmt", "yuv420p", "-r", str(FPS), str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_clips(clips: list[Path], out_path: Path, work: Path) -> None:
    listfile = work / "concat.txt"
    listfile.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
           "-c:v", VCODEC, "-b:v", VBITRATE, "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)


# ── The streaming pipeline ────────────────────────────────────────────────────
async def run_pipeline(req: StoryRequest):
    loop = asyncio.get_event_loop()

    # ---- 1. article text ----------------------------------------------------
    article = (req.text or "").strip()
    if not article and req.url:
        yield progress("fetch", "Fetching article…", pct=3)
        try:
            article = await fetch_article(req.url)
        except Exception as ex:
            yield sse({"error": f"Could not fetch the URL ({type(ex).__name__}). "
                                "Paste the article text instead."})
            return
    if len(article) < 200:
        yield sse({"error": "Article text too short — paste the full article."})
        return
    yield progress("fetch", f"Article ready ({len(article.split())} words)", pct=8)

    # ---- 2. storyboard ------------------------------------------------------
    if req.storyboard:
        storyboard = req.storyboard
    else:
        yield progress("storyboard", "Drafting storyboard (Gemma)…", pct=12)
        try:
            storyboard = await make_storyboard(article, req.n_scenes, req.anonymize)
        except Exception as ex:
            yield sse({"error": f"Storyboard step failed: {type(ex).__name__}: {ex}"})
            return
    n = len(storyboard)
    yield sse({"storyboard": {"scenes": storyboard, "n": n}})
    yield progress("storyboard", f"Storyboard ready — {n} scenes", total=n, pct=18)

    if req.mode == "storyboard":          # preview-only: stop before the slow part
        yield sse({"done": True})
        return

    work = Path(tempfile.mkdtemp(prefix="story_"))
    try:
        # ---- 3. voiceover per scene -----------------------------------------
        wavs: list[Path | None] = []
        any_audio = False
        for i, sc in enumerate(storyboard, 1):
            yield progress("voice", f"Voicing narration {i}/{n} (Chatterbox)…",
                           step=i, total=n, pct=18 + int(22 * i / n))
            audio = await tts_scene(sc["narration"])
            if audio:
                wp = work / f"voice_{i:02d}.wav"
                wp.write_bytes(audio)
                wavs.append(wp)
                any_audio = True
            else:
                wavs.append(None)
        if not any_audio:
            yield progress("voice", "TTS unavailable — building a silent cut "
                           "(wire /tts on the Ditto service to add narration)",
                           total=n, pct=40)

        # ---- 4. image per scene ---------------------------------------------
        imgs: list[Path] = []
        for i, sc in enumerate(storyboard, 1):
            yield progress("image", f"Generating image {i}/{n} (Flux)…",
                           step=i, total=n, pct=40 + int(40 * i / n))
            # Option 1: one global art style applied uniformly to every scene,
            # appended to whatever visual description Gemma wrote, so all N
            # scenes share a consistent look.
            scene_prompt = sc["image_prompt"]
            if req.style:
                scene_prompt = f"{scene_prompt}. Art style: {req.style}"
            data = await flux_scene(scene_prompt)
            ip = work / f"scene_{i:02d}.png"
            if data:
                ip.write_bytes(data)
                imgs.append(ip)
                thumb = "data:image/png;base64," + base64.b64encode(data).decode()
                yield progress("image", f"Scene {i}/{n} ready", step=i, total=n,
                               pct=40 + int(40 * i / n), thumb=thumb)
            else:
                yield sse({"error": f"Flux failed on scene {i}. Is its SLURM job up?"})
                return

        # ---- 5. render ------------------------------------------------------
        clips = []
        for i, (ip, wp) in enumerate(zip(imgs, wavs), 1):
            yield progress("render", f"Rendering scene {i}/{n}…",
                           step=i, total=n, pct=80 + int(15 * i / n))
            dur = wav_duration(wp) if wp else NO_AUDIO_SCENE_SECS
            clip = work / f"clip_{i:02d}.mp4"
            try:
                await loop.run_in_executor(None, render_scene_clip, ip, wp, clip, dur)
                clips.append(clip)
            except subprocess.CalledProcessError as ex:
                err = ex.stderr.decode()[-300:] if ex.stderr else str(ex)
                yield sse({"error": f"ffmpeg failed on scene {i}: {err}"})
                return

        yield progress("render", "Stitching final video…", pct=97)
        final = work / "story.mp4"
        try:
            await loop.run_in_executor(None, concat_clips, clips, final, work)
        except subprocess.CalledProcessError as ex:
            err = ex.stderr.decode()[-300:] if ex.stderr else str(ex)
            yield sse({"error": f"ffmpeg concat failed: {err}"})
            return

        b64 = base64.b64encode(final.read_bytes()).decode()
        yield sse({"generated_video": b64, "num_frames": 0,
                   "prompt": f"Visual story · {n} scenes", "model": "Story"})
        yield sse({"done": True})
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/ready")
def ready():
    return {"ready": True}


@app.post("/story")
async def story(req: StoryRequest):
    return StreamingResponse(run_pipeline(req), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
