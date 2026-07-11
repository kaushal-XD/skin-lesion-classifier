🧬 Prompt: Skin Lesion Classification Web App
> Ready-to-paste prompt for Google AI Studio / Gemini
---
PROMPT (paste this directly into Google AI Studio)
---
You are an expert full-stack web developer. Build a single-file HTML web application for skin lesion classification. The app must be fully self-contained — all HTML, CSS, and JavaScript in one `.html` file with no build tools required.
---
🎯 PROJECT OVERVIEW
A prototype medical-grade web tool that:
Lets the user select one of 6 pre-trained skin lesion classification models hosted on Hugging Face
Accepts an uploaded skin lesion image
Runs inference via the Hugging Face Inference API and returns the top 3 predictions with confidence scores
For each of those 3 predicted disease names, calls the Google Gemini API to generate a structured disease info card
Provides an interactive chat interface so the user can ask follow-up questions about those diseases
---
🤖 MODELS (Hugging Face — all by KaushalXD)
Use the HuggingFace Inference API endpoint:
```
POST https://api-inference.huggingface.co/models/<MODEL_ID>
Headers: { Authorization: "Bearer <HF_TOKEN>", Content-Type: "application/json" }
Body: send image as base64 binary blob
```
The 6 available models and their display names:
Display Name	Hugging Face Model ID
DenseNet121 (Focal Loss v2)	`KaushalXD/DenseNet121-focal-loss-v2`
DenseNet121 (Focal Loss)	`KaushalXD/DenseNet121-focal-loss`
DenseNet121	`KaushalXD/skin-lesion-densenet121`
MobileNetV2	`KaushalXD/skin-lesion-mobileNetv2`
ResNet18	`KaushalXD/skin-lesion-resnet18`
ResNet50	`KaushalXD/skin-lesion-resnet50`
HF Inference API returns an array like:
```json
[
  { "label": "melanoma", "score": 0.87 },
  { "label": "nevus",    "score": 0.09 },
  ...
]
```
Extract the top 3 results by score.
---
💊 DISEASE INFO CARDS (via Gemini API)
After getting top 3 predictions, call Gemini API once per prediction to generate a disease info card.
Gemini API call:
```
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=<GEMINI_KEY>
Body: { "contents": [{ "parts": [{ "text": "<your prompt>" }] }] }
```
Prompt template for each disease card:
```
You are a medical information assistant. The user has uploaded a skin lesion image and the AI model predicted it might be "{DISEASE_NAME}" with {CONFIDENCE}% confidence.

Generate a concise, structured medical info card for "{DISEASE_NAME}" in the following JSON format only, no markdown:

{
  "disease": "<full proper disease name>",
  "overview": "<2-3 sentence plain-language overview>",
  "symptoms": ["<symptom 1>", "<symptom 2>", "<symptom 3>", "<symptom 4>"],
  "causes": ["<cause 1>", "<cause 2>", "<cause 3>"],
  "risk_factors": ["<risk factor 1>", "<risk factor 2>", "<risk factor 3>"],
  "precautions": ["<precaution 1>", "<precaution 2>", "<precaution 3>"],
  "when_to_see_doctor": "<1-2 sentence guidance>",
  "severity": "Low | Medium | High",
  "is_contagious": true | false
}

Return ONLY valid JSON. No extra text.
```
Parse the JSON response and render each disease as a visual info card.
---
💬 INTERACTIVE CHAT
Below the 3 info cards, render a simple chat interface:
System context (sent silently with every message):
```
  You are a helpful dermatology assistant. The user's skin image was analyzed and the top predictions were: {disease1} ({conf1}%), {disease2} ({conf2}%), {disease3} ({conf3}%). Answer questions about these conditions clearly and concisely. Always recommend professional medical consultation for diagnosis.
  ```
User types a question → sent to Gemini API with full chat history maintained in JS array
Render assistant replies below each user message in a scrollable chat window
Show a "typing..." indicator while waiting for Gemini response
Include a disclaimer: "This tool is for educational purposes only. It is not a substitute for professional medical diagnosis."
---
🖥️ UI / UX REQUIREMENTS
Layout — Three-Panel Flow (top to bottom):
Panel 1 — Configuration
App title: "SkinSight — Lesion Classifier"
Subtitle: "AI-powered skin lesion analysis prototype"
Row with two inputs side by side:
Dropdown: "Select Model" with all 6 models listed
Text input: "Hugging Face API Token" (type=password)
Text input: "Gemini API Key" (type=password)
API keys are stored only in JS memory (never sent anywhere except the respective APIs)
Panel 2 — Image Upload & Inference
Large drag-and-drop upload zone (also supports click-to-browse)
After upload: show image preview (max 300px height)
"Analyze Image" button — triggers HF inference call
Loading spinner / progress indicator while model loads (HF cold-starts can take 20–60s; show a message: "Model may take up to 60 seconds to warm up on first request...")
After inference: display top 3 results as horizontal confidence bars with disease label and percentage score
Panel 3 — Disease Info Cards
Three side-by-side cards (stack vertically on mobile)
Each card shows:
Disease name + rank badge (#1, #2, #3)
Severity badge (color-coded: Low=green, Medium=orange, High=red)
Contagious badge if applicable
Overview text
Four sections with icons: Symptoms 🔴, Causes ⚡, Precautions 🛡️, Risk Factors ⚠️
Each section lists items as clean bullet pills
"When to see a doctor" highlighted box at bottom
Show skeleton loader cards while Gemini is generating
Panel 4 — Chat
Chat window (fixed height, scrollable)
Input bar with send button
Renders user messages right-aligned, assistant messages left-aligned
Clear conversation button
Visual Style:
Clean, clinical white/light-gray background
Accent color: deep teal `#0d7377` for buttons and active elements
Card shadows for depth, rounded corners (12px)
Monospace font for confidence scores, sans-serif for body text
Fully responsive (mobile-friendly)
No external CSS frameworks — write all CSS inline or in a `<style>` block
---
⚠️ ERROR HANDLING
Handle the following gracefully with user-visible messages:
Missing API keys → show inline warning on button click
HF model still loading (HTTP 503) → show retry countdown (poll every 10s, max 3 retries)
HF returns unexpected format → show "Model returned unexpected output. Please try again."
Gemini rate limit or error → show "Could not load disease info. Try again later." inside the card
Invalid image format → "Please upload a JPG, PNG, or WEBP image."
Network failure → generic "Connection error. Check your internet and try again."
---
🔐 IMPORTANT CONSTRAINTS
API keys are entered by the user at runtime and stored only in JavaScript variables — never hardcoded, never logged to console
No backend, no server — 100% browser-side
All API calls go directly from browser to Hugging Face / Google APIs
Add a CORS note comment in code: HF Inference API supports browser requests; Gemini API supports browser requests with API key
The Gemini JSON response may be wrapped in markdown fences — strip `json and` before `JSON.parse()`
---
📁 OUTPUT FORMAT
Return a single complete `.html` file with:
`<!DOCTYPE html>` declaration
All CSS in a `<style>` block in `<head>`
All JavaScript in a `<script>` block before `</body>`
No external dependencies except Google Fonts (one import for typography)
Fully functional when opened locally in a browser
The file should be production-ready with clean, well-commented code organized in logical sections.
---
✅ QUICK CHECKLIST (verify before outputting)
[ ] All 6 models are listed in the dropdown with correct HF model IDs
[ ] HF Inference API called with image as binary/base64 and Bearer token
[ ] Top 3 predictions extracted and displayed as confidence bars
[ ] Gemini called once per top-3 disease, JSON parsed correctly
[ ] Disease cards render all 7 fields from the JSON schema
[ ] Chat maintains history and sends system context with every message
[ ] Typing indicator shown while waiting for Gemini
[ ] All error states handled
[ ] Responsive layout
[ ] Medical disclaimer visible
---
This is a prototype for academic/research demonstration. All medical information is AI-generated and must not replace professional diagnosis.