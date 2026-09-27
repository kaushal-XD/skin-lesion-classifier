"""
SkinSight Backend Server
Pre-loads DenseNet121-focal-loss-v2 model at startup for fast inference.
Disease cards are loaded locally from skin_disease_reference_guide.json;
only the follow-up chat is proxied through OpenRouter.
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
HF_MODEL_ID   = "KaushalXD/ViT-Large-focal-loss_v1"
HF_MODEL_NAME = "ViT-Large (Focal Loss v1)"
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EasyCLIProxyAPI config — local proxy running at http://127.0.0.1:8317
# Key is server-side only, never exposed to the browser
# Load from .env file directly so it works even if python-dotenv is not installed
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())
    except Exception as e:
        print(f"[SkinSight] Could not read .env: {e}")

# Read the API key from .env (the "First key" shown in EasyCLIProxyAPI dashboard)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "123456")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "123456"

# EasyCLIProxyAPI exposes an OpenAI-compatible API at /v1
OPENROUTER_URL  = "http://127.0.0.1:8317/v1/chat/completions"

# Primary model — Claude is best for medical context with Antigravity OAuth
OPENROUTER_MODEL = "claude-sonnet-4-6"          # primary
# Fallback chain — tried in order if primary is rate-limited (429) or unavailable (404)
# These are the EXACT model IDs returned by http://127.0.0.1:8317/v1/models
OPENROUTER_FALLBACKS = [
    "claude-opus-4-6-thinking",
    "gemini-3.8-flash-high",
    "gemini-3.7-flash-high",
    "gemini-3.6-flash-high",
    "gemini-3-flash",
    "gemini-3.1-flash-image",
    "gemini-pro-agent",
    "gemini-3.1-pro-low",
    "gpt-oss-120b-medium",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
]
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
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

REFERENCE_GUIDE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skin_disease_reference_guide.json",
)
with open(REFERENCE_GUIDE_PATH, "r", encoding="utf-8") as guide_file:
    _reference_guide = json.load(guide_file)


def normalize_label(label: str) -> str:
    """Normalize model labels so they match reference-guide class names."""
    return "".join(character.lower() for character in label if character.isalnum())


_reference_by_label = {
    normalize_label(entry["class_name"]): entry
    for entry in _reference_guide["classes"]
}
print(f"[SkinSight] Loaded {len(_reference_by_label)} disease reference entries.")

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
# DISEASE INFO CARD  (loaded from the local reference guide)
# ============================================================================
class CardRequest(BaseModel):
    label: str
    confidence: float


@app.post("/api/openai/card")
async def openai_card(req: CardRequest):
    """Return the matching disease entry without making an external API call."""
    entry = _reference_by_label.get(normalize_label(req.label))
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No reference information found for model label '{req.label}'.",
        )

    return {
        **entry,
        "predicted_label": req.label,
        "confidence": req.confidence,
        "disclaimer": _reference_guide["disclaimer"],
    }


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