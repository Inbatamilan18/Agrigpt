"""
main.py — AgriGPT FastAPI server
================================
Endpoints:
  POST /api/signup   create account (email + password + farm details)
  POST /api/login    login -> JWT token
  GET  /api/me       current user
  PUT  /api/profile  update language / state / main crop
  GET  /api/history  chat history (memory)
  POST /api/chat     the main endpoint: text and/or image -> advice
  GET  /api/health   liveness check

The frontend (frontend/) is served from the same server at "/",
so the browser opens http://localhost:8000 and everything is same-origin.

Run:  uvicorn main:app --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # must run before other agriGPT imports

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from auth import Message, User, create_access_token, decode_token, get_db, hash_password, verify_password
from rag import build_or_load_vectorstore, generate_advice, translate_to_english
from vision import predict_disease

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("agrigpt")

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(_BASE_DIR, "frontend")

limiter = Limiter(key_func=get_remote_address)
bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AgriGPT...")
    try:
        build_or_load_vectorstore()
        logger.info("Knowledge base ready (RAG).")
    except Exception as exc:  # noqa: BLE001
        logger.error("Knowledge base init failed: %s", exc)
    logger.info("Open http://localhost:8000 in your browser.")
    yield


app = FastAPI(title="AgriGPT", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a minute and try again."},
    )


# ---------------------------- request models --------------------------------

class AuthIn(BaseModel):
    email: str
    password: str
    language: str = "en"
    state: str = ""
    main_crop: str = ""


class ProfileIn(BaseModel):
    language: Optional[str] = None
    state: Optional[str] = None
    main_crop: Optional[str] = None


# ---------------------------- helpers ----------------------------------------

def _valid_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[-1]


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing token")
    payload = decode_token(creds.credentials)
    user = db.query(User).filter(User.email == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _public_user(u: User) -> dict:
    return {
        "email": u.email,
        "language": u.language,
        "state": u.state or "",
        "main_crop": u.main_crop or "",
    }


# ---------------------------- endpoints --------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "app": "AgriGPT"}


@app.post("/api/signup")
@limiter.limit("5/minute")
def signup(request: Request, data: AuthIn, db: Session = Depends(get_db)):
    if not _valid_email(data.email):
        raise HTTPException(400, "Enter a valid email address")
    if len(data.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "This email is already registered")
    user = User(
        email=data.email,
        hashed_password=hash_password(data.password),
        language=data.language or "en",
        state=data.state or "",
        main_crop=data.main_crop or "",
    )
    db.add(user)
    db.commit()
    return {"access_token": create_access_token({"sub": user.email}), "user": _public_user(user)}


@app.post("/api/login")
@limiter.limit("10/minute")
def login(request: Request, data: AuthIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(401, "Wrong email or password")
    return {"access_token": create_access_token({"sub": user.email}), "user": _public_user(user)}


@app.get("/api/me")
def me(user: User = Depends(get_current_user)):
    return _public_user(user)


@app.put("/api/profile")
def update_profile(data: ProfileIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if data.language is not None:
        user.language = data.language
    if data.state is not None:
        user.state = data.state
    if data.main_crop is not None:
        user.main_crop = data.main_crop
    db.commit()
    return _public_user(user)


@app.get("/api/history")
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Message)
        .filter(Message.user_email == user.email)
        .order_by(Message.id.desc())
        .limit(50)
        .all()
    )
    return {"messages": [{"role": r.role, "content": r.content} for r in reversed(rows)]}


@app.post("/api/chat")
@limiter.limit("20/minute")
def chat(
    request: Request,
    message: str = Form(""),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    image: Optional[UploadFile] = File(None),
):
    has_text = bool(message and message.strip())
    if not has_text and image is None:
        raise HTTPException(400, "Send a message or upload an image")

    # 1) Disease detection from the photo (if provided)
    disease = ""
    confidence = None
    if image is not None:
        raw = image.file.read()
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(400, "Image too large (max 10 MB)")
        try:
            res = predict_disease(raw)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        disease, confidence = res["disease"], res["confidence"]

    # 2) Language handling: Hindi query -> English for retrieval
    lang = user.language or "en"
    query = message.strip() if has_text else (
        "What disease is shown in this crop image and how should I treat it?"
    )
    query = translate_to_english(query)

    # 3) Recent conversation (the "memory")
    recent = (
        db.query(Message)
        .filter(Message.user_email == user.email)
        .order_by(Message.id.desc())
        .limit(6)
        .all()
    )
    history_text = "\n".join(
        f"{'Farmer' if m.role == 'user' else 'AgriGPT'}: {m.content}" for m in reversed(recent)
    )

    # 4) Farmer profile personalisation
    profile = ""
    if user.state or user.main_crop:
        profile = (
            f"\nFARMER PROFILE: region = {user.state or 'unknown'}, "
            f"main crop = {user.main_crop or 'unknown'}. Personalise advice to this farmer."
        )

    # 5) RAG + LLM
    advice = generate_advice(query, disease, lang, history_text, profile)

    # 6) Save the conversation
    db.add(Message(user_email=user.email, role="user", content=query or f"[image: {disease}]"))
    db.add(Message(user_email=user.email, role="assistant", content=advice))
    db.commit()

    return {"advice": advice, "disease": disease, "confidence": confidence}


# Serve the frontend at "/" (API routes above take priority)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
