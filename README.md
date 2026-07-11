# 🔬 SkinSight — Skin Lesion Classifier

> AI-powered skin lesion analysis prototype using deep learning and LLM-generated medical information cards.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

SkinSight is a web-based skin lesion classification tool that:

- **Classifies** skin lesion images using a pre-trained **DenseNet121** model (fine-tuned with focal loss)
- **Generates** structured disease information cards via free LLMs (OpenRouter)
- **Provides** an interactive chat interface to ask follow-up questions about identified conditions

> ⚠️ **Disclaimer**: This tool is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Browser (index.html)            │
│  Upload Image → View Results → Chat          │
└──────────────────┬──────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────┐
│           FastAPI Backend (server.py)         │
│                                              │
│  /api/classify  → Local DenseNet121 model     │
│  /api/openai/card → OpenRouter (free LLMs)    │
│  /api/openai/chat → OpenRouter (free LLMs)    │
└──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/skin-lesion-classifier.git
cd skin-lesion-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
# Get a free key at https://openrouter.ai/keys
```

Or set the environment variable directly:

```bash
# Windows
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Linux/macOS
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 4. Run the server

```bash
python server.py
```

The app will be available at **http://localhost:8001**

> 💡 On first run, the DenseNet121 model weights (~30MB) will be downloaded automatically from Hugging Face.

## 🧠 Model

The classification model is **DenseNet121 (Focal Loss v2)** fine-tuned on skin lesion datasets:

| Model | Hugging Face ID |
|-------|----------------|
| DenseNet121 (Focal Loss v2) | [`KaushalXD/DenseNet121-focal-loss-v2`](https://huggingface.co/KaushalXD/DenseNet121-focal-loss-v2) |

The model classifies skin lesions into categories including melanoma, nevus, basal cell carcinoma, and more.

## 🔧 Tech Stack

- **Frontend**: Single-file HTML/CSS/JS (no build tools required)
- **Backend**: Python FastAPI with Uvicorn
- **ML**: PyTorch + HuggingFace Transformers
- **LLM**: OpenRouter API (free tier — DeepSeek, Llama, Gemma fallback chain)

## 📁 Project Structure

```
skin-lesion-classifier/
├── server.py           # FastAPI backend — model inference + LLM proxy
├── index.html          # Frontend — single-file web UI
├── project.md          # Original project specification
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.
