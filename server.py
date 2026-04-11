"""Vercel (and other ASGI hosts): entry that exposes the FastAPI ``app``.

See https://vercel.com/docs/frameworks/backend/fastapi
Local development typically uses ``uvicorn app.main:app`` instead.
"""

from app.main import app

__all__ = ["app"]
