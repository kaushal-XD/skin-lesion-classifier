"""
SkinSight Backend Server
Pre-loads DenseNet121-focal-loss-v2 model at startup for fast inference.
LLM calls (disease cards + chat) are proxied through OpenRouter — free, no user key needed.
"""

import io
import os
import sys
import json

# Force UTF-8 output — prevents Windows 'charmap' codec crash when printing
# non-ASCII characters (emojis etc.) to the console.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import requests as http_requests
from pydantic import BaseModel
from typing import List
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoModelForImageClassification
from torchvision import transforms

# ============================================================================
# CONFIG
# ============================================================================
HF_MODEL_ID   = "KaushalXD/DenseNet121-focal-loss-v2"
HF_MODEL_NAME = "DenseNet121 (Focal Loss v2)"
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# OpenRouter config  — key is server-side only, never exposed to the browser
# Load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional — user can set env vars manually

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "deepseek/deepseek-v4-flash:free"          # primary (large context, fast)
# Fallback chain — tried in order if primary is rate-limited (429) or unavailable (404)
OPENROUTER_FALLBACKS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-coder:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8001",   # required by OpenRouter
    "X-Title": "SkinSight",
}

# Standard ImageNet preprocessing (used by DenseNet121)
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================================
# PRE-LOAD MODEL AT STARTUP
# ============================================================================
print(f"[SkinSight] Using device: {DEVICE}")
print(f"[SkinSight] Pre-loading model: {HF_MODEL_ID}...")
print(f"[SkinSight] This may take 30-60 seconds on first run (downloading weights)...")

_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
_model.to(DEVICE)
_model.eval()

print(f"[SkinSight] Model loaded and ready!")

# ============================================================================
# HELPERS
# ============================================================================

def openrouter_chat(messages: list, temperature: float = 0.3) -> str:
    """
    Call OpenRouter chat completions with automatic fallback.
    Tries OPENROUTER_MODEL first, then each model in OPENROUTER_FALLBACKS.
    """
    models_to_try = [OPENROUTER_MODEL] + OPENROUTER_FALLBACKS
    last_error = ""

    for model in models_to_try:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        try:
            resp = http_requests.post(
                url=OPENROUTER_URL,
                headers=OPENROUTER_HEADERS,
                data=json.dumps(payload),
                timeout=60,
            )
        except Exception as e:
            last_error = str(e)
            print(f"[SkinSight] Request error with {model}: {e}")
            continue

        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"].get("content", "")
            print(f"[SkinSight] Used model: {model}")
            return content

        # 404 = model unavailable, 429 = rate-limited → try next
        if resp.status_code in (404, 429):
            print(f"[SkinSight] {model} returned {resp.status_code}, trying fallback...")
            last_error = resp.text[:200]
            continue

        # Any other error — fail immediately
        print(f"[SkinSight] OpenRouter error {resp.status_code}: {resp.text[:300]}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter API error {resp.status_code}: {resp.text[:300]}"
        )

    # All models exhausted
    raise HTTPException(
        status_code=502,
        detail=f"All OpenRouter free models are currently unavailable. Last error: {last_error[:200]}"
    )


def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences that models sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text

# ============================================================================
# APP
# ============================================================================
app = FastAPI(title="SkinSight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================================
# CLASSIFY
# ============================================================================
@app.post("/api/classify")
async def classify_image(image: UploadFile = File(...)):
    """Classify a skin lesion image. Returns all predictions sorted by confidence."""
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = _transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = _model(pixel_values=input_tensor)

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        id2label = _model.config.id2label

        results = [
            {"label": id2label.get(idx, f"class_{idx}"), "score": float(prob)}
            for idx, prob in enumerate(probs)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"predictions": results, "model": HF_MODEL_NAME}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# STATUS
# ============================================================================
@app.get("/api/status")
async def status():
    """Check if server and model are ready."""
    return {
        "status": "ready",
        "model": HF_MODEL_NAME,
        "model_id": HF_MODEL_ID,
        "device": str(DEVICE),
        "llm": OPENROUTER_MODEL,
    }


# ============================================================================
# DISEASE INFO CARD  (via OpenRouter — no user API key required)
# ============================================================================
class CardRequest(BaseModel):
    label: str
    confidence: float


@app.post("/api/openai/card")
async def openai_card(req: CardRequest):
    """Generate a disease info card via OpenRouter (free LLM)."""
    prompt = (
        f'You are a medical information assistant. The user has uploaded a skin lesion image '
        f'and the AI model predicted it might be "{req.label}" with {req.confidence:.1f}% confidence.\n\n'
        f'Generate a concise, structured medical info card for "{req.label}" '
        f'in the following JSON format ONLY — no markdown, no preamble:\n\n'
        '{\n'
        '  "disease": "<full proper disease name>",\n'
        '  "overview": "<2-3 sentence plain-language overview>",\n'
        '  "symptoms": ["<symptom 1>", "<symptom 2>", "<symptom 3>", "<symptom 4>"],\n'
        '  "causes": ["<cause 1>", "<cause 2>", "<cause 3>"],\n'
        '  "risk_factors": ["<risk factor 1>", "<risk factor 2>", "<risk factor 3>"],\n'
        '  "precautions": ["<precaution 1>", "<precaution 2>", "<precaution 3>"],\n'
        '  "when_to_see_doctor": "<1-2 sentence guidance>",\n'
        '  "severity": "Low | Medium | High",\n'
        '  "is_contagious": true | false\n'
        '}\n\nReturn ONLY valid JSON. No extra text.'
    )

    try:
        text = openrouter_chat([{"role": "user", "content": prompt}], temperature=0.3)
        text = strip_json_fences(text)
        print(f"[SkinSight] Card response for '{req.label}':\n{text[:300]}")
        return json.loads(text)

    except json.JSONDecodeError as e:
        print(f"[SkinSight] JSON parse error: {e}\nRaw: {text[:300]}")
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {e}")
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CHAT  (via OpenRouter — no user API key required)
# ============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    system_prompt: str
    history: List[ChatMessage]


@app.post("/api/openai/chat")
async def openai_chat(req: ChatRequest):
    """Proxy chat messages to OpenRouter (free LLM)."""
    try:
        messages = [{"role": "system", "content": req.system_prompt}]
        messages += [{"role": m.role, "content": m.content} for m in req.history]

        reply = openrouter_chat(messages, temperature=0.7)
        return {"reply": reply or "Sorry, I could not generate a response."}

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SERVE FRONTEND
# ============================================================================
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)