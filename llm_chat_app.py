"""
Gemma 4 multimodal chat app.
Supports text + image uploads + audio/video (via Whisper transcription), streaming responses.
Run via serve_llm.slurm or: uvicorn llm_chat_app:app --port 8766
"""
import asyncio, base64, io, json, os, re, tempfile, threading
from pathlib import Path

import httpx
import torch
import uvicorn
import whisper as _whisper_lib
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, TextIteratorStreamer

MODEL_PATH   = os.environ.get("MODEL_PATH",   "/scratch/users/t07an25/llm_experiments/gemma4")
WHISPER_PATH = os.environ.get("WHISPER_PATH", "/scratch/users/t07an25/llm_experiments/whisper")
PORT         = int(os.environ.get("PORT", 8766))

# Image-generation services (separate SLURM jobs, separate MIG slices)
FLUX_GEN_URL  = os.environ.get("FLUX_GEN_URL",  "http://gpu02:8768")   # Flux.1 schnell
VIDEO_GEN_URL = os.environ.get("VIDEO_GEN_URL", "http://gpu02:8769")   # Wan2.1 1.3B
TALK_GEN_URL  = os.environ.get("TALK_GEN_URL",  "http://gpu02:8770")   # Ditto + Chatterbox
STORY_GEN_URL = os.environ.get("STORY_GEN_URL", "http://gpu02:8772")   # Visual-story orchestrator

# System prompt prepended to every conversation. Override at runtime with
# the SYSTEM_PROMPT env var (e.g. in serve_llm.slurm) or edit the default below.
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You are Gemma, a helpful, concise multimodal assistant. "
    "You can see images and read transcripts of audio. "
    "Format answers in clean Markdown — use headings, bullet lists and code blocks where useful. "
    "If you are unsure, say so."
))

# Long transcripts are split into chunks; each chunk is summarised individually,
# then the summaries are combined for the final Gemma 4 response.
CHUNK_WORDS           = 900   # words per chunk sent to Gemma 4
LONG_TRANSCRIPT_WORDS = 1200  # transcripts longer than this use chunked mode

# OOM controls.
MAX_HISTORY_TURNS     = 6     # keep last N user+model exchanges (12 messages total)
MAX_IMAGE_EDGE        = 896   # downscale images so longest edge ≤ this
MAX_INPUT_TOKENS_SOFT = 6000  # if prompt exceeds this, drop oldest history

app = FastAPI(title="Gemma 4 Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model         = None
_processor     = None
_whisper_model = None
_lock          = threading.Lock()   # one inference at a time

# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model():
    global _model, _processor, _whisper_model
    print(f"[startup] Loading Gemma 4 from {MODEL_PATH} …", flush=True)
    _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    _model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    print("[startup] Gemma 4 ready.", flush=True)
    print(f"[startup] Loading Whisper medium from {WHISPER_PATH} …", flush=True)
    _whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[startup] Loading Whisper medium on {_whisper_device} …", flush=True)
    _whisper_model = _whisper_lib.load_model("medium", download_root=WHISPER_PATH, device=_whisper_device)
    print("[startup] Whisper ready.", flush=True)

def _generate_text_sync(msgs: list, max_new_tokens: int = 256) -> str:
    """Non-streaming single-shot generation — used for per-chunk summarisation."""
    with _lock:
        inputs = _processor.apply_chat_template(
            msgs, add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = _model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return _processor.tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)

# ── HTML frontend ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 Chat</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --accent: #7c6af7; --accent2: #5b4fcf; --text: #e8eaf6;
    --text-dim: #8892b0; --border: #2d3152; --user-bg: #1e2a4a;
    --ai-bg: #1a1d27; --danger: #e05c5c; --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }

  /* Header */
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  .logo { width: 32px; height: 32px; background: linear-gradient(135deg, var(--accent), #a78bfa); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
  header h1 { font-size: 1rem; font-weight: 600; }
  header span { font-size: .78rem; color: var(--text-dim); margin-left: auto; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; display: inline-block; margin-right: 6px; }
  .status-dot.loading { background: #facc15; animation: pulse 1s infinite; }

  /* Chat area */
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth; }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  /* Messages */
  .msg { display: flex; gap: 10px; max-width: 85%; animation: fadeIn .2s ease; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg.ai   { align-self: flex-start; }
  .avatar { width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 14px; }
  .msg.user .avatar { background: var(--accent2); }
  .msg.ai   .avatar { background: var(--surface2); }
  .bubble { padding: 10px 14px; border-radius: var(--radius); line-height: 1.6; font-size: .9rem; max-width: 100%; }
  .msg.user .bubble { background: var(--user-bg); border-bottom-right-radius: 3px; }
  .msg.ai   .bubble { background: var(--ai-bg); border: 1px solid var(--border); border-bottom-left-radius: 3px; }
  .bubble img { max-width: 280px; border-radius: 8px; margin-bottom: 6px; display: block; }
  .bubble pre { background: #0d0f18; border-radius: 6px; padding: 10px; overflow-x: auto; font-size: .82rem; margin: 8px 0; }
  .bubble code { font-family: 'Cascadia Code', 'Fira Mono', monospace; font-size: .83rem; background: #0d0f18; padding: 1px 4px; border-radius: 3px; }
  .bubble pre code { background: none; padding: 0; }
  /* Markdown elements inside AI bubbles */
  .bubble p { margin: 0 0 8px; }
  .bubble p:last-child { margin-bottom: 0; }
  .bubble h1, .bubble h2, .bubble h3, .bubble h4, .bubble h5, .bubble h6 {
    margin: 14px 0 6px; font-weight: 600; line-height: 1.3;
  }
  .bubble h1 { font-size: 1.25rem; }
  .bubble h2 { font-size: 1.15rem; }
  .bubble h3 { font-size: 1.05rem; }
  .bubble h4, .bubble h5, .bubble h6 { font-size: .95rem; }
  .bubble ul, .bubble ol { margin: 4px 0 8px; padding-left: 22px; }
  .bubble li { margin: 2px 0; }
  .bubble li > p { margin: 0; }
  .bubble blockquote { margin: 6px 0; padding: 6px 12px; border-left: 3px solid var(--accent); background: rgba(124,106,247,.07); color: var(--text-dim); border-radius: 4px; }
  .bubble a { color: #a78bfa; text-decoration: underline; }
  .bubble a:hover { color: #c4b5fd; }
  .bubble table { border-collapse: collapse; margin: 8px 0; font-size: .82rem; }
  .bubble th, .bubble td { border: 1px solid var(--border); padding: 4px 8px; text-align: left; }
  .bubble th { background: var(--surface2); font-weight: 600; }
  .bubble hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }
  .bubble strong { font-weight: 600; color: #fff; }
  .bubble em { font-style: italic; }

  /* Typing indicator */
  .typing { display: flex; gap: 4px; align-items: center; padding: 12px 14px; }
  .typing span { width: 7px; height: 7px; background: var(--text-dim); border-radius: 50%; animation: bounce .9s infinite; }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }

  /* Input bar */
  #inputbar { background: var(--surface); border-top: 1px solid var(--border); padding: 14px 20px; flex-shrink: 0; }
  #preview-area { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
  .preview-item { position: relative; }
  .preview-item img { width: 60px; height: 60px; object-fit: cover; border-radius: 8px; border: 1px solid var(--border); }
  .preview-item audio { height: 36px; max-width: 220px; border-radius: 8px; }
  .preview-item .fname { font-size:.75rem; color:var(--text-dim); max-width:160px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .preview-item .remove { position: absolute; top: -6px; right: -6px; background: var(--danger); border: none; color: #fff; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center; }
  .input-row { display: flex; gap: 8px; align-items: flex-end; }
  #msg-input { flex: 1; background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); font-size: .9rem; padding: 10px 14px; resize: none; outline: none; min-height: 44px; max-height: 140px; line-height: 1.5; font-family: inherit; }
  #msg-input:focus { border-color: var(--accent); }
  #msg-input::placeholder { color: var(--text-dim); }
  .btn { padding: 10px 14px; border: none; border-radius: var(--radius); cursor: pointer; font-size: .88rem; transition: opacity .15s; display: flex; align-items: center; gap: 6px; }
  .btn:hover { opacity: .85; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn-upload { background: var(--surface2); color: var(--text-dim); border: 1px solid var(--border); }
  .btn-send   { background: var(--accent); color: #fff; font-weight: 600; }

  /* Welcome */
  #welcome { text-align: center; margin: auto; color: var(--text-dim); }
  #welcome .big { font-size: 2.5rem; margin-bottom: 8px; }
  #welcome h2 { font-size: 1.1rem; font-weight: 500; margin-bottom: 4px; color: var(--text); }
  #welcome p  { font-size: .85rem; }

  @keyframes fadeIn  { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
  @keyframes bounce  { 0%,80%,100% { transform:scale(.8); } 40% { transform:scale(1.2); } }
  @keyframes pulse   { 0%,100% { opacity:1; } 50% { opacity:.4; } }
</style>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
</head>
<body>

<header>
  <div class="logo">✦</div>
  <h1>Gemma 4 Chat</h1>
  <span><span class="status-dot loading" id="sdot"></span><span id="slabel">Loading model…</span></span>
</header>

<div id="chat">
  <div id="welcome">
    <div class="big">✦</div>
    <h2>Gemma 4 Multimodal Chat</h2>
    <p>Send text, upload images, transcribe audio.<br>
       <code>/imageflux &lt;prompt&gt;</code> — generate an image (Flux.1 schnell, ~85s).<br>
       <code>/video &lt;prompt&gt;</code> — generate a 5 s video clip (Wan2.1 1.3B, ~8 min).<br>
       <code>/talk &lt;text&gt;</code> — talking head video. Optionally attach a face photo and/or an audio clip (5-20 s) to clone that voice.<br>
       <code>/storyboard &lt;url|text&gt;</code> — preview the scene breakdown of an article (fast).<br>
       <code>/story [--style "art style"] [--vertical] &lt;url|text&gt;</code> — full narrated visual story with live progress (~several min). Optional <code>--style</code> applies one look to every scene (e.g. <code>--style "watercolour storybook"</code>); <code>--vertical</code> (or <code>--aspect 9:16</code>) renders Instagram/Reels portrait.</p>
  </div>
</div>

<div id="inputbar">
  <div id="preview-area"></div>
  <div class="input-row">
    <button class="btn btn-upload" onclick="document.getElementById('file-input').click()" title="Upload image, audio or video">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
    </button>
    <input type="file" id="file-input" multiple accept="image/*,audio/*,video/*,.mp3,.wav,.m4a,.mp4,.webm,.ogg" style="display:none" onchange="handleFile(this)">
    <textarea id="msg-input" placeholder="Message Gemma 4…" rows="1"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}"
      oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,140)+'px'"></textarea>
    <button class="btn btn-send" id="send-btn" onclick="send()" disabled>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </div>
</div>

<script>
let history = [];
let pendingImage = null;   // { file, dataUrl }      — picture for vision / face
let pendingAudio = null;   // { file, dataUrl, name } — audio for Whisper / voice-clone
let ready = false;

// Poll until model ready
async function pollReady() {
  try {
    const r = await fetch('/ready');
    if (r.ok && (await r.json()).ready) {
      ready = true;
      document.getElementById('sdot').className = 'status-dot';
      document.getElementById('slabel').textContent = 'Gemma 4 ready';
      document.getElementById('send-btn').disabled = false;
      return;
    }
  } catch {}
  setTimeout(pollReady, 2000);
}
pollReady();

function handleFile(input) {
  // Accept one or more files at once; dispatch each into the image or audio slot.
  const files = Array.from(input.files || []);
  files.forEach(file => {
    const isAudio = file.type.startsWith('audio/') || file.type.startsWith('video/') ||
      /\.(mp3|wav|m4a|mp4|webm|ogg|flac|aac)$/i.test(file.name);
    const reader = new FileReader();
    reader.onload = e => {
      if (isAudio) pendingAudio = { file, dataUrl: e.target.result, name: file.name };
      else         pendingImage = { file, dataUrl: e.target.result };
      renderPreviews();
    };
    reader.readAsDataURL(file);
  });
  input.value = '';
}

function renderPreviews() {
  const area = document.getElementById('preview-area');
  area.innerHTML = '';
  if (pendingImage) {
    const w = document.createElement('div');
    w.className = 'preview-item';
    w.innerHTML = `<img src="${pendingImage.dataUrl}"><button class="remove" onclick="clearImage()">✕</button>`;
    area.appendChild(w);
  }
  if (pendingAudio) {
    const w = document.createElement('div');
    w.className = 'preview-item';
    w.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;background:var(--surface2);padding:6px 10px;border-radius:8px;border:1px solid var(--border)">
        <span style="font-size:1.2rem">🎵</span>
        <span class="fname">${esc(pendingAudio.name)}</span>
        <button class="remove" onclick="clearAudio()" style="position:static;margin-left:4px">✕</button>
      </div>`;
    area.appendChild(w);
  }
}

function clearImage() { pendingImage = null; renderPreviews(); }
function clearAudio() { pendingAudio = null; renderPreviews(); }

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Configure marked once for GitHub-flavoured markdown + auto line breaks
if (window.marked) {
  marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });
}

function renderMarkdown(text) {
  // While streaming, an unclosed ``` fence makes marked render everything
  // after it as a code block. Add a temporary closing fence if needed.
  let safe = text;
  const fenceCount = (text.match(/```/g) || []).length;
  if (fenceCount % 2 === 1) safe = text + '\n```';
  try {
    const html = window.marked ? marked.parse(safe) : esc(safe).replace(/\n/g, '<br>');
    return window.DOMPurify ? DOMPurify.sanitize(html) : html;
  } catch (e) {
    return esc(text).replace(/\n/g, '<br>');
  }
}

function appendMsg(role, text, imgUrl, audUrl, audName) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const avatar = role === 'user' ? '👤' : '✦';
  // Back-compat: callers that pass a single mediaUrl + boolean still work.
  if (typeof audUrl === 'boolean') {
    const wasAudio = audUrl;
    audUrl = wasAudio ? imgUrl : null;
    imgUrl = wasAudio ? null   : imgUrl;
    audName = null;
  }
  let mediaHtml = '';
  if (imgUrl) {
    mediaHtml += `<img src="${imgUrl}" alt="uploaded image">`;
  }
  if (audUrl) {
    mediaHtml += `<audio controls src="${audUrl}" style="max-width:260px;display:block;margin:6px 0"></audio>`;
    if (audName) mediaHtml += `<div style="font-size:.7rem;color:var(--text-dim);margin-bottom:6px">🎵 ${esc(audName)}</div>`;
  }
  div.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="bubble" id="bubble-${Date.now()}">
      ${mediaHtml}
      <span class="text-content">${role==='user' ? esc(text) : renderMarkdown(text)}</span>
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div.querySelector('.text-content');
}

function appendTranscript(transcript) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ai';
  // Split into paragraphs on double newlines; fall back to sentence grouping
  const paras = transcript.split(/\n\n+/).map(p => p.trim()).filter(Boolean);
  const html = paras.length > 1
    ? paras.map(p => `<p style="margin:0 0 6px">${esc(p)}</p>`).join('')
    : `<p style="margin:0;white-space:pre-wrap">${esc(transcript)}</p>`;
  div.innerHTML = `
    <div class="avatar">📝</div>
    <div class="bubble" style="border-color:#4a5568;opacity:.85;max-height:260px;overflow-y:auto">
      <div style="font-size:.75rem;color:var(--text-dim);margin-bottom:6px;font-weight:600">🎙️ Whisper transcript</div>
      ${html}
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function appendGeneratedVideo(b64data, prompt, model, numFrames) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ai';
  const fps = 16; const dur = numFrames ? (numFrames / fps).toFixed(1) + 's' : '';
  const modelLabel = model ? ` · ${esc(model)}` : '';
  const videoUrl = 'data:video/mp4;base64,' + b64data;
  div.innerHTML = `
    <div class="avatar">🎬</div>
    <div class="bubble">
      <div style="font-size:.75rem;color:var(--text-dim);margin-bottom:6px;font-weight:600">🎬 Generated video${modelLabel}${dur ? ' · ' + dur : ''}</div>
      <video controls autoplay loop muted playsinline
        style="max-width:512px;width:100%;border-radius:8px;display:block"
        src="${videoUrl}"></video>
      <div style="font-size:.78rem;color:var(--text-dim);margin-top:6px;font-style:italic">"${esc(prompt)}"</div>
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function appendGeneratedImage(dataUrl, prompt, model) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ai';
  const modelLabel = model ? ` · ${esc(model)}` : '';
  div.innerHTML = `
    <div class="avatar">🎨</div>
    <div class="bubble">
      <div style="font-size:.75rem;color:var(--text-dim);margin-bottom:6px;font-weight:600">🎨 Generated image${modelLabel}</div>
      <img src="${dataUrl}" alt="generated image" style="max-width:512px;width:100%;border-radius:8px;display:block">
      <div style="font-size:.78rem;color:var(--text-dim);margin-top:6px;font-style:italic">"${esc(prompt)}"</div>
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

// ── Visual-story progress bubble ──────────────────────────────────────────
// One bubble that mutates in place as the orchestrator streams events:
//   progress  → stage list + bar + ETA + per-scene thumbnails
//   storyboard→ expandable scene list (narration + image prompt)
//   the final generated_video is rendered by appendGeneratedVideo()
let storyEl = null;       // the live bubble's inner container
let storyThumbs = [];     // accumulated scene thumbnails

const STAGE_ORDER = ['init','fetch','storyboard','voice','image','render'];
const STAGE_LABEL = { init:'Start', fetch:'Article', storyboard:'Storyboard',
                      voice:'Narration', image:'Images', render:'Render' };

function ensureStoryBubble() {
  if (storyEl) return storyEl;
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ai';
  div.innerHTML = `
    <div class="avatar">📖</div>
    <div class="bubble" style="min-width:280px">
      <div style="font-size:.75rem;color:var(--text-dim);margin-bottom:8px;font-weight:600">📖 Building visual story…</div>
      <div id="story-bar-wrap" style="background:var(--surface2);border-radius:6px;height:8px;overflow:hidden;margin-bottom:4px">
        <div id="story-bar" style="height:100%;width:1%;background:linear-gradient(90deg,var(--accent),#a78bfa);transition:width .3s"></div>
      </div>
      <div id="story-eta" style="font-size:.7rem;color:var(--text-dim);margin-bottom:8px"></div>
      <div id="story-stages" style="font-size:.8rem;line-height:1.7"></div>
      <div id="story-thumbs" style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px"></div>
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  storyEl = div;
  storyThumbs = [];
  return div;
}

function updateStoryProgress(p) {
  const el = ensureStoryBubble();
  const bar = el.querySelector('#story-bar');
  if (typeof p.pct === 'number') bar.style.width = Math.max(1, Math.min(100, p.pct)) + '%';
  const eta = el.querySelector('#story-eta');
  if (p.eta_s) eta.textContent = '~' + Math.ceil(p.eta_s / 60) + ' min left';
  // Stage checklist — current stage shows its label, earlier stages get a tick
  const curIdx = STAGE_ORDER.indexOf(p.stage);
  const lines = STAGE_ORDER.map((s, i) => {
    if (i < curIdx) return `<div>✓ ${STAGE_LABEL[s]}</div>`;
    if (i === curIdx) return `<div style="color:var(--text)">⟳ ${esc(p.label || STAGE_LABEL[s])}`
        + (p.total ? ` <span style="color:var(--text-dim)">(${p.step}/${p.total})</span>` : '') + `</div>`;
    return `<div style="color:var(--text-dim)">· ${STAGE_LABEL[s]}</div>`;
  });
  el.querySelector('#story-stages').innerHTML = lines.join('');
  if (p.thumb) {
    storyThumbs.push(p.thumb);
    el.querySelector('#story-thumbs').innerHTML =
      storyThumbs.map(t => `<img src="${t}" style="width:48px;height:28px;object-fit:cover;border-radius:3px">`).join('');
  }
  document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
}

function appendStoryboard(sb) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ai';
  const rows = (sb.scenes || []).map(s =>
    `<div style="margin:6px 0;padding:6px 8px;background:var(--surface2);border-radius:6px">
       <div style="font-weight:600">Scene ${s.n}</div>
       <div style="margin:2px 0">${esc(s.narration || '')}</div>
       <div style="font-size:.75rem;color:var(--text-dim);font-style:italic">🎨 ${esc(s.image_prompt || '')}</div>
     </div>`).join('');
  div.innerHTML = `
    <div class="avatar">🗂️</div>
    <div class="bubble">
      <div style="font-size:.75rem;color:var(--text-dim);margin-bottom:6px;font-weight:600">🗂️ Storyboard — ${sb.n} scenes</div>
      ${rows}
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function appendTyping() {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ai';
  div.id = 'typing-indicator';
  div.innerHTML = `<div class="avatar">✦</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function send() {
  if (!ready) return;
  const input = document.getElementById('msg-input');
  const text = input.value.trim();
  if (!text && !pendingImage && !pendingAudio) return;

  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  const imgUrl   = pendingImage ? pendingImage.dataUrl : null;
  const audUrl   = pendingAudio ? pendingAudio.dataUrl : null;
  const audName  = pendingAudio ? pendingAudio.name    : null;
  const imgFile  = pendingImage ? pendingImage.file    : null;
  const audFile  = pendingAudio ? pendingAudio.file    : null;
  appendMsg('user', text, imgUrl, audUrl, audName);

  let typingEl = appendTyping();
  let aiTextEl = null;
  let fullText = '';
  storyEl = null;   // start a fresh story-progress bubble for this turn

  const fd = new FormData();
  fd.append('message', text);
  fd.append('history', JSON.stringify(history));
  if (imgFile) fd.append('image', imgFile);
  if (audFile) fd.append('audio', audFile);
  clearImage();
  clearAudio();

  try {
    const resp = await fetch('/chat', { method: 'POST', body: fd });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));
        if (data.done) break;
        if (data.error) {
          if (typingEl) typingEl.remove();
          appendMsg('ai', `Error: ${data.error}`, null, false);
          btn.disabled = false;
          return;
        }
        if (data.status) {
          const icon = data.status.startsWith('Summarising') ? '🧩' : '🎙️';
          typingEl.querySelector('.bubble').innerHTML =
            `<span style="font-size:.8rem;color:var(--text-dim)">${icon} ${esc(data.status)}</span>`;
          continue;
        }
        if (data.transcript) {
          appendTranscript(data.transcript);
          // Reset typing indicator for Gemma response
          typingEl.remove();
          typingEl = appendTyping();
          continue;
        }
        if (data.progress) {
          if (typingEl) { typingEl.remove(); typingEl = null; }
          updateStoryProgress(data.progress);
          continue;
        }
        if (data.storyboard) {
          appendStoryboard(data.storyboard);
          continue;
        }
        if (data.generated_video) {
          if (typingEl) typingEl.remove();
          if (storyEl) { storyEl = null; }   // finalise the live bubble
          appendGeneratedVideo(data.generated_video, data.prompt, data.model, data.num_frames);
          continue;
        }
        if (data.generated_image) {
          typingEl.remove();
          appendGeneratedImage(data.generated_image, data.prompt, data.model);
          continue;
        }
        if (!aiTextEl) {
          typingEl.remove();
          aiTextEl = appendMsg('ai', '', null, false);
        }
        fullText += data.text;
        aiTextEl.innerHTML = renderMarkdown(fullText);
        document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
      }
    }
  } catch (e) {
    if (typingEl) typingEl.remove();
    appendMsg('ai', `Connection error: ${e.message}`, null);
  }

  // Update history (text only — images not stored in history)
  if (text) history.push({ role: 'user', content: [{ type: 'text', text }] });
  if (fullText) history.push({ role: 'model', content: [{ type: 'text', text: fullText }] });

  btn.disabled = false;
  document.getElementById('msg-input').focus();
}
</script>
</body>
</html>"""

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return HTMLResponse(HTML)

@app.get("/ready")
def ready():
    return {"ready": _model is not None}

@app.post("/generate_text")
async def generate_text(payload: dict = Body(...)):
    """Single-shot text generation for internal services (e.g. the story
    orchestrator's storyboard step). Reuses the already-loaded Gemma model
    under the same inference lock as /chat."""
    if _model is None:
        return {"error": "Model not loaded yet"}
    prompt = (payload.get("prompt") or "").strip()
    system = (payload.get("system") or "").strip()
    max_new = int(payload.get("max_new_tokens", 512))
    if not prompt:
        return {"error": "prompt is required"}
    msgs = []
    if system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(
            None, lambda: _generate_text_sync(msgs, max_new_tokens=max_new)
        )
        return {"text": text}
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}

@app.post("/chat")
async def chat(
    message: str = Form(""),
    history: str = Form("[]"),
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    if _model is None:
        async def err():
            yield f'data: {json.dumps({"error": "Model not loaded yet"})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    # Read uploads eagerly — can't await inside the generator
    audio_bytes    = await audio.read() if (audio and audio.filename) else None
    audio_suffix   = (Path(audio.filename).suffix or ".audio") if (audio and audio.filename) else None
    image_bytes    = await image.read() if (image and image.filename) else None
    msg_text       = message.strip()
    use_fp16       = torch.cuda.is_available()

    try:
        hist = json.loads(history)
    except Exception:
        hist = []

    async def stream_response():
        loop = asyncio.get_event_loop()

        # ── Slash commands ────────────────────────────────────────────────────
        # /imageflux <prompt>  → Flux.1 schnell image (~85s)
        # /video <prompt>      → Wan2.1 1.3B video (~8 min)
        # /talk <text>         → Ditto talking head video (~3–5 min)
        #                        Attach a face photo to use your own face,
        #                        or set TALK_FACE_PATH on the server as fallback.
        # ── /story | /storyboard — relay the orchestrator's SSE stream ────────
        # Unlike the other slash commands (single POST → one result), the story
        # pipeline is long and multi-stage, so we proxy its event stream line by
        # line. Progress / storyboard / video events pass straight through to the
        # browser, which is the whole point: live, granular progress.
        story_cmd = None
        if msg_text.lower().startswith("/storyboard "):
            story_cmd = ("storyboard", msg_text[len("/storyboard "):].strip())
        elif msg_text.lower().startswith("/story "):
            story_cmd = ("render", msg_text[len("/story "):].strip())

        if story_cmd is not None:
            mode, arg = story_cmd
            # Optional global art style: --style "watercolour storybook" (quoted,
            # multi-word) or --style noir (single token). Applied to every scene's
            # image prompt. Strip it out so what's left is the URL / article text.
            story_style = None
            m = re.search(r'--style\s+"([^"]+)"|--style\s+\'([^\']+)\'|--style\s+(\S+)', arg)
            if m:
                story_style = (m.group(1) or m.group(2) or m.group(3)).strip()
                arg = (arg[:m.start()] + arg[m.end():]).strip()
            # Optional aspect ratio: --aspect 9:16 (or 1:1 / 4:5 / 16:9), or the
            # shorthand --vertical for 9:16 Instagram/Reels/TikTok format.
            story_aspect = None
            ma = re.search(r'--aspect\s+(\S+)', arg)
            if ma:
                story_aspect = ma.group(1).strip()
                arg = (arg[:ma.start()] + arg[ma.end():]).strip()
            mv = re.search(r'(?:^|\s)--vertical(?:\s|$)', arg)
            if mv:
                story_aspect = story_aspect or "9:16"
                arg = (arg[:mv.start()] + " " + arg[mv.end():]).strip()
            if not arg:
                _usage = json.dumps({"error": 'Usage: /story [--style "art style"] [--aspect 9:16|1:1|4:5] [--vertical] <article URL or pasted text>'})
                yield f"data: {_usage}\n\n"
                return
            # A leading http(s):// is treated as a URL to fetch; anything else is
            # treated as the article text pasted directly into the chat.
            body = {"mode": mode}
            if story_style:
                body["style"] = story_style
            if story_aspect:
                body["aspect"] = story_aspect
            if re.match(r"^https?://", arg, re.I):
                body["url"] = arg
            else:
                body["text"] = arg
            yield f'data: {json.dumps({"progress": {"stage": "init", "label": "Starting visual story…", "pct": 1}})}\n\n'
            try:
                async with httpx.AsyncClient(timeout=1800.0) as client:
                    async with client.stream("POST", f"{STORY_GEN_URL}/story", json=body) as r:
                        if r.status_code != 200:
                            await r.aread()
                            yield f'data: {json.dumps({"error": f"Story service error: {r.text[:200]}"})}\n\n'
                            return
                        async for line in r.aiter_lines():
                            if line.startswith("data: "):
                                yield line + "\n\n"
                return
            except httpx.ConnectError:
                yield f'data: {json.dumps({"error": f"Story service unreachable at {STORY_GEN_URL}. Is its SLURM job running?"})}\n\n'
                return
            except Exception as ex:
                import traceback; print(f'[Story error] {type(ex).__name__}: {ex}\n{traceback.format_exc()}', flush=True)
                yield f'data: {json.dumps({"error": f"Story pipeline failed: {type(ex).__name__}: {ex}"})}\n\n'
                return

        slash = None
        if msg_text.lower().startswith("/imageflux "):
            slash = ("flux",  FLUX_GEN_URL,  "Flux",   msg_text[len("/imageflux "):].strip())
        elif msg_text.lower().startswith("/video "):
            slash = ("video", VIDEO_GEN_URL, "Wan2.1", msg_text[len("/video "):].strip())
        elif msg_text.lower().startswith("/talk "):
            slash = ("talk",  TALK_GEN_URL,  "Ditto",  msg_text[len("/talk "):].strip())

        if slash is not None:
            kind, base_url, label, prompt = slash
            if not prompt:
                cmd_names = {"video": "video", "talk": "talk", "flux": "imageflux"}
                yield f'data: {json.dumps({"error": f"Usage: /{cmd_names.get(kind, kind)} <prompt>"})}\n\n'
                return
            yield f'data: {json.dumps({"status": f"Generating with {label}: {prompt[:60]}…"})}\n\n'
            try:
                # Timeouts: Ditto ~600s, Wan2.1 ~900s, Flux ~300s
                if kind == "talk":  timeout = 600.0
                elif kind == "video": timeout = 900.0
                elif kind == "flux":  timeout = 300.0
                else: timeout = 120.0

                # Build request — /talk can accept an uploaded face image
                # and/or an uploaded audio clip used as the voice reference for
                # Chatterbox cloning (overrides server-side TALK_VOICE_PATH).
                req_body: dict = {"prompt": prompt}
                if kind == "talk" and image_bytes:
                    req_body["face_image"] = base64.b64encode(image_bytes).decode()
                if kind == "talk" and audio_bytes:
                    req_body["voice_ref"] = base64.b64encode(audio_bytes).decode()

                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(f"{base_url}/generate", json=req_body)
                if r.status_code != 200:
                    yield f'data: {json.dumps({"error": f"{label} error: {r.text[:200]}"})}\n\n'
                    return
                data = r.json()
                if "error" in data:
                    yield f'data: {json.dumps({"error": data["error"]})}\n\n'
                    return
                if kind in ("video", "talk"):
                    yield f'data: {json.dumps({"generated_video": data["video"], "num_frames": data.get("num_frames", 0), "prompt": prompt, "model": label})}\n\n'
                else:
                    yield f'data: {json.dumps({"generated_image": data["image"], "prompt": prompt, "model": label})}\n\n'
                yield f'data: {json.dumps({"done": True})}\n\n'
                return
            except httpx.ConnectError:
                yield f'data: {json.dumps({"error": f"{label} service unreachable at {base_url}. Is its SLURM job running?"})}\n\n'
                return
            except Exception as ex:
                import traceback; print(f'[{label} error] {type(ex).__name__}: {ex}\n{traceback.format_exc()}', flush=True)
                yield f'data: {json.dumps({"error": f"{label} generation failed: {type(ex).__name__}: {ex}"})}\n\n'
                return

        # ── Step 1: transcribe audio non-blocking ─────────────────────────────
        transcript = None
        if audio_bytes:
            yield f'data: {json.dumps({"status": "transcribing"})}\n\n'
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                result = await loop.run_in_executor(
                    None,
                    lambda: _whisper_model.transcribe(tmp_path, fp16=use_fp16)
                )
                transcript = result["text"].strip()
                n_words = len(transcript.split())
                print(f"[whisper] transcript ({n_words} words): {transcript[:120]}…", flush=True)
                yield f'data: {json.dumps({"transcript": transcript})}\n\n'
            except Exception as ex:
                yield f'data: {json.dumps({"error": f"Transcription failed: {ex}"})}\n\n'
                return
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # ── Step 2: build Gemma 4 message content ────────────────────────────
        content = []
        if image_bytes:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Downscale to keep vision-tower KV cache manageable
            if max(pil_img.size) > MAX_IMAGE_EDGE:
                pil_img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
                print(f"[image] downscaled to {pil_img.size}", flush=True)
            content.append({"type": "image", "image": pil_img})

        context_text = ""
        if transcript:
            words = transcript.split()
            if len(words) > LONG_TRANSCRIPT_WORDS:
                # ── Chunked summarisation ──────────────────────────────────
                chunks = [
                    " ".join(words[i: i + CHUNK_WORDS])
                    for i in range(0, len(words), CHUNK_WORDS)
                ]
                total = len(chunks)
                summaries = []
                for idx, chunk in enumerate(chunks, 1):
                    yield f'data: {json.dumps({"status": f"Summarising segment {idx}/{total}…"})}\n\n'
                    chunk_msgs = [{"role": "user", "content": [{"type": "text", "text":
                        f"Concisely summarise this transcript segment (segment {idx} of {total}):\n\n{chunk}"
                    }]}]
                    try:
                        summary = await loop.run_in_executor(
                            None, lambda m=chunk_msgs: _generate_text_sync(m, max_new_tokens=220)
                        )
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        summary = f"[Segment {idx} — OOM during summarisation]"
                    summaries.append(f"Segment {idx}/{total}: {summary}")
                    print(f"[chunk {idx}/{total}] {summary[:80]}…", flush=True)

                combined_summaries = "\n\n".join(summaries)
                context_text = (
                    f"[Audio transcript — {total} segments summarised]\n\n"
                    f"{combined_summaries}\n\n"
                )
            else:
                # Short enough to send directly
                context_text = f"[Audio transcript]:\n{transcript}\n\n"

        combined_text = context_text
        if msg_text:
            combined_text += msg_text
        elif transcript:
            combined_text += "Please provide a comprehensive summary of this audio content."

        if combined_text:
            content.append({"type": "text", "text": combined_text})

        if not content:
            yield f'data: {json.dumps({"done": True})}\n\n'
            return

        # Trim history: keep last MAX_HISTORY_TURNS user+model pairs.
        # Use a local rebind to avoid Python's UnboundLocalError on `hist`.
        trimmed = hist
        if len(trimmed) > MAX_HISTORY_TURNS * 2:
            dropped = len(trimmed) - MAX_HISTORY_TURNS * 2
            trimmed = trimmed[-MAX_HISTORY_TURNS * 2:]
            print(f"[history] trimmed {dropped} old messages (kept last {len(trimmed)})", flush=True)

        msgs = []
        if SYSTEM_PROMPT:
            msgs.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        msgs.extend({"role": t.get("role", "user"), "content": t.get("content", [])} for t in trimmed)
        msgs.append({"role": "user", "content": content})

        # Token-budget guard: if the prompt is still huge after history trim,
        # drop more old messages until we fit (preserves system context + current turn).
        def _prompt_tokens(m):
            try:
                ids = _processor.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_tensors="pt")
                return ids.shape[-1]
            except Exception:
                return 0

        while len(msgs) > 1 and _prompt_tokens(msgs) > MAX_INPUT_TOKENS_SOFT:
            removed = msgs.pop(0)
            print(f"[token-budget] dropped {removed.get('role')} message ({_prompt_tokens(msgs)} tokens remaining)", flush=True)

        # ── Step 3: stream Gemma 4 inference ─────────────────────────────────
        streamer = TextIteratorStreamer(
            _processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inference_error: list = []   # mutable container so the thread can write into it

        def run_inference(max_new_tokens=1024, retry=False):
            with _lock:
                try:
                    inputs = _processor.apply_chat_template(
                        msgs, add_generation_prompt=True,
                        tokenize=True, return_tensors="pt", return_dict=True,
                    )
                    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
                    _model.generate(
                        **inputs, streamer=streamer,
                        max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=0.7, top_p=0.9,
                    )
                except torch.cuda.OutOfMemoryError as ex:
                    print(f"[OOM] {ex}", flush=True)
                    torch.cuda.empty_cache()
                    if not retry:
                        print("[OOM] retrying with max_new_tokens=256", flush=True)
                        # Recreate streamer for the retry attempt
                        run_inference(max_new_tokens=256, retry=True)
                    else:
                        inference_error.append(
                            "GPU out of memory even after retry. Try clearing the chat "
                            "(refresh the page) and sending a shorter message or a smaller image."
                        )
                except Exception as ex:
                    inference_error.append(str(ex))
                    print(f"[inference error] {ex}", flush=True)
                finally:
                    streamer.end()

        thread = threading.Thread(target=run_inference, daemon=True)
        thread.start()

        queue = asyncio.Queue()
        def feed():
            for chunk in streamer:
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=feed, daemon=True).start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield f'data: {json.dumps({"text": chunk})}\n\n'

        if inference_error:
            yield f'data: {json.dumps({"error": inference_error[0]})}\n\n'
        else:
            yield f'data: {json.dumps({"done": True})}\n\n'

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
