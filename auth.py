"""
auth.py — Authentication & database layer for AgriGPT
=====================================================
- SQLite database (users + chat history) via SQLAlchemy
- Password hashing with bcrypt (passlib)
- JWT tokens with python-jose

This is the "security" part examiners ask about:
every API call from the frontend must carry a valid JWT.
"""

import os
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = os.getenv("SECRET_KEY", "agrigpt-dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

Base = declarative_base()
engine = create_engine(
    f"sqlite:///{os.path.join(_BASE_DIR, 'agrigpt.db')}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    language = Column(String, default="en")          # "en" or "hi"
    state = Column(String, default="")               # farmer's region
    main_crop = Column(String, default="")           # used to personalise advice
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Message(Base):
    __tablename__ = "messages"                       # chat history = "memory"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    role = Column(String)                            # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
