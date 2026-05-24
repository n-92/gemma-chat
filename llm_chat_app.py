"""
Gemma 4 multimodal chat app.
Supports text + image uploads + audio/video (via Whisper transcription), streaming responses.
Run via serve_llm.slurm or: uvicorn llm_chat_app:app --port 8766
"""
import asyncio, base64, io, json, os, tempfile, threading
from pathlib import Path

import torch
import uvicorn
import whisper as _whisper_lib
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, TextIteratorStreamer

MODEL_PATH   = os.environ.get("MODEL_PATH",   "/scratch/users/t07an25/llm_experiments/gemma4")
WHISPER_PATH = os.environ.get("WHISPER_PATH", "/scratch/users/t07an25/llm_experiments/whisper")
PORT         = int(os.environ.get("PORT", 8766))

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
    <p>Send text, upload images — ask anything.</p>
  </div>
</div>

<div id="inputbar">
  <div id="preview-area"></div>
  <div class="input-row">
    <button class="btn btn-upload" onclick="document.getElementById('file-input').click()" title="Upload image, audio or video">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
    </button>
    <input type="file" id="file-input" accept="image/*,audio/*,video/*,.mp3,.wav,.m4a,.mp4,.webm,.ogg" style="display:none" onchange="handleFile(this)">
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
let pendingImage = null;   // { file, dataUrl }
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
  const file = input.files[0];
  if (!file) return;
  const isAudio = file.type.startsWith('audio/') || file.type.startsWith('video/') ||
    /\.(mp3|wav|m4a|mp4|webm|ogg|flac|aac)$/i.test(file.name);
  const reader = new FileReader();
  reader.onload = e => {
    pendingImage = { file, dataUrl: e.target.result, isAudio };
    const area = document.getElementById('preview-area');
    area.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'preview-item';
    if (isAudio) {
      wrap.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;background:var(--surface2);padding:6px 10px;border-radius:8px;border:1px solid var(--border)">
          <span style="font-size:1.2rem">🎵</span>
          <span class="fname">${esc(file.name)}</span>
          <button class="remove" onclick="clearImage()" style="position:static;margin-left:4px">✕</button>
        </div>`;
    } else {
      wrap.innerHTML = `<img src="${e.target.result}"><button class="remove" onclick="clearImage()">✕</button>`;
    }
    area.appendChild(wrap);
  };
  reader.readAsDataURL(file);
  input.value = '';
}

function clearImage() {
  pendingImage = null;
  document.getElementById('preview-area').innerHTML = '';
}

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

function appendMsg(role, text, mediaUrl, isAudio) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const avatar = role === 'user' ? '👤' : '✦';
  let mediaHtml = '';
  if (mediaUrl && isAudio) {
    mediaHtml = `<audio controls src="${mediaUrl}" style="max-width:260px;display:block;margin-bottom:6px"></audio>`;
  } else if (mediaUrl) {
    mediaHtml = `<img src="${mediaUrl}" alt="uploaded image">`;
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
  if (!text && !pendingImage) return;

  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  const mediaUrl   = pendingImage ? pendingImage.dataUrl : null;
  const isAudio    = pendingImage ? pendingImage.isAudio : false;
  const pendingFile = pendingImage ? pendingImage.file : null;
  appendMsg('user', text, mediaUrl, isAudio);

  let typingEl = appendTyping();
  let aiTextEl = null;
  let fullText = '';

  const fd = new FormData();
  fd.append('message', text);
  fd.append('history', JSON.stringify(history));
  if (pendingFile) {
    if (isAudio) fd.append('audio', pendingFile);
    else         fd.append('image', pendingFile);
  }
  clearImage();

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
