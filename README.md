# 🌾 AgriGPT — Multimodal RAG Agriculture Advisor (MVP)

A chatbot that works like a 24×7 Kisan Call Center: the farmer types a question,
speaks it, or uploads a photo of a diseased crop — and gets practical,
document-grounded advice in English or Hindi.

**Built with:** FastAPI · Llama 3.1 (Groq) · RAG (sentence-transformers + LangChain + Chroma) ·
pre-trained ViT (PlantVillage) · JWT auth (bcrypt + SQLite) · SlowAPI rate limiting · NLLB-200 (Hindi) ·
HTML/CSS/JS frontend.

---

## 📁 Project structure

```
agrigpt/
├── main.py                  # FastAPI server, all /api endpoints, serves the frontend
├── auth.py                  # JWT login, bcrypt hashing, SQLite (users + chat history)
├── rag.py                   # RAG pipeline: docs → embeddings → Chroma → LLM context
├── vision.py                # Disease detection from leaf photos (pre-trained ViT)
├── download_models.py       # One-time model downloader
├── requirements.txt
├── .env.example             # → copy to .env and add your Groq key
├── data/agriculture_docs/   # 6 knowledge documents that power the RAG
│   ├── tomato_diseases.txt
│   ├── potato_diseases.txt
│   ├── rice_paddy_diseases.txt
│   ├── wheat_maize_diseases.txt
│   ├── cotton_fruit_veg_diseases.txt
│   └── soil_fertilizer_organic.txt
└── frontend/
    ├── index.html           # Login/Signup + chat UI
    ├── style.css
    └── script.js            # JWT handling, image upload, voice in/out
```

---

## ⏱️ 5-HOUR COMPLETION PLAN

| Time | Step | What happens |
|---|---|---|
| 0:00–0:30 | **Step 1** | Install Python deps (takes longest, start it first) |
| 0:30–0:45 | **Step 2** | Get free Groq API key + create `.env` |
| 0:45–1:45 | **Step 3** | `python download_models.py` (model downloads) |
| 1:45–2:30 | **Step 4** | Start server, sign up, test text chat |
| 2:30–3:30 | **Step 5** | Test image upload + voice + Hindi |
| 3:30–4:30 | **Step 6** | Take screenshots / record demo video |
| 4:30–5:00 | **Step 7** | Push to GitHub + optional Railway deploy |

> Need: Python 3.10+, ~8 GB free disk, decent internet (Chrome or Edge for the demo).

### Step 1 — Install (start this FIRST, it's the slowest part)

Windows:
```bat
cd agrigpt
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
macOS / Linux:
```bash
cd agrigpt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(On Windows the extra CPU-only torch line saves ~2 GB. On Mac/Linux plain pip torch is fine.)

### Step 2 — Groq key + .env

1. Go to **https://console.groq.com/keys** → sign in (free) → **Create API Key**.
2. In the `agrigpt` folder:
   - Windows: `copy .env.example .env`
   - Mac/Linux: `cp .env.example .env`
3. Open `.env`, paste your key: `GROQ_API_KEY=gsk_....`

### Step 3 — Download models (one time, ~1 GB; ~15-30 min)

```bash
python download_models.py
```
If you want to demo **Hindi** too (extra ~2.5 GB):
```bash
python download_models.py --with-nllb
```

### Step 4 — Run the server

```bash
uvicorn main:app --port 8000
```
You should see: `Knowledge base ready (RAG)` and `Open http://localhost:8000`.

Open **http://localhost:8000** in Chrome/Edge → **Sign up** (enter your state + main crop —
this powers the personalisation) → ask something, e.g.:
- *"My tomato leaves have dark patches and fruits are turning black"*
- *"How much fertilizer for 1 acre of paddy?"*
- *"What is zinc deficiency in wheat?"*

### Step 5 — Test the multimodal features

- **📷 Image:** click the camera button, upload a leaf photo (healthy tomato / diseased rice /
  wheat leaf — any from PlantVillage or your own). The bot shows a disease chip
  (e.g. `Tomato_Late_blight — 97%`) and treatment advice.
- **🎤 Voice:** click the mic (Chrome/Edge), speak your question.
- **🔊 Voice reply:** answers are spoken aloud automatically (browser TTS).
- **हिंदी:** switch the EN/हिंदी dropdown at top-right (works fully if you ran `--with-nllb`;
  answers stay in Hindi even without it).
- **Memory:** ask a follow-up — the bot remembers the last 6 messages.

### Step 6 — Demo screenshots / video

Suggested demo script (2-3 min):
1. Show the login page → sign up with your farm details (10 s)
2. Text question about a disease → show the structured RAG answer (30 s)
3. Upload a diseased leaf photo → show disease chip + treatment (40 s)
4. Ask the same in Hindi or switch language (20 s)
5. Voice question + voice answer (20 s)
6. Show the terminal: JWT auth, rate limiting, Chroma retrieval logs (15 s)

### Step 7 — Push to GitHub (+ optional Railway deploy)

```bash
git init && git add -A && git commit -m "AgriGPT MVP"
```
Create the repo on GitHub and push. (`.env`, `chroma_db/`, `agrigpt.db` are already git-ignored.)

Railway/Render deploy: new project from the repo → build `pip install -r requirements.txt` →
start `uvicorn main:app --host 0.0.0.0 --port $PORT` → add `GROQ_API_KEY` + `SECRET_KEY` as env vars.
(First deploy re-downloads models, so the first chat request is slow — that's expected.)

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `bcrypt` / `passlib` import error at startup | `pip install bcrypt==4.0.1` |
| `port 8000 already in use` | `uvicorn main:app --port 8001` and open that port |
| First chat is very slow | Normal — models load on first use. Second request is fast |
| `401 / invalid token` error after restart | Just login again in the browser |
| Groq `401` | Wrong key in `.env` — must start with `gsk_` |
| Groq `404 model not found` | Open console.groq.com/models, set `LLM_MODEL` to an available model name in `.env` |
| Groq `rate limit` (429) | Free tier limit — wait a minute, or switch `LLM_MODEL` to `llama-3.1-70b-versatile` |
| Hindi reply falls back to English | You didn't run `download_models.py --with-nllb` |
| Bot seems to ignore a new document you added | Delete the `chroma_db` folder once, restart the server (re-embeds) |
| `torch` download too big / too slow | Use the CPU-only index (Windows line in Step 1) |

---

## 🧠 How it works (Viva one-liners)

- **RAG:** documents → 800-char chunks → sentence-transformer embeddings → Chroma vector DB →
  top-4 chunks injected into the LLM prompt → grounded, hallucination-controlled answers.
- **Multimodal:** photo → ViT (PlantVillage-finetuned) → disease label + confidence → label is
  fed to RAG + LLM → treatment advice specific to that disease.
- **Security:** bcrypt password hashing, JWT on every API call, per-user SQLite history,
  SlowAPI rate limits (20 chat req/min, 5 signup/min).
- **Memory:** last 6 messages + farmer profile (state, main crop) are included in every prompt.
- **Multilingual:** Hindi query → NLLB-200 → English for retrieval → LLM answers in Hindi.

## 🚀 Upgrade path to the FULL final-year version

| MVP (this) | Final version |
|---|---|
| Pre-trained ViT | Fine-tuned **YOLOv8s** on PlantVillage (Colab notebook — ask for it) |
| MiniLM embeddings | **BGE-M3** (`EMBEDDING_MODEL` line in `.env`) |
| Hindi + English | **8 Indian languages** via NLLB (Tamil, Telugu, Marathi, Bengali, Kannada, Gujarati, Punjabi, Odia) |
| Browser TTS | **Whisper + Coqui TTS** server-side |
| Local run | **Railway/Render + NGINX reverse proxy + UFW + HTTPS** |
| 6 .txt docs | Real **ICAR PDFs** added to `data/agriculture_docs/` |
