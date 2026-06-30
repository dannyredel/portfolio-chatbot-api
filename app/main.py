from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.chat import answer_stream
from app.config import settings

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Portfolio Chatbot API")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


# --- Global daily cost guardrail ---------------------------------------------
# A hard ceiling on total /api/chat requests per UTC day, across all users. The
# per-IP rate limit above can be bypassed by rotating IPs; this caps the total
# blast radius. In-memory (resets on restart / per process) — a safety net, not
# the hard wall. The real hard wall is the monthly spend limit set on the
# ANTHROPIC and OPENAI API keys in their provider consoles.
_daily = {"date": None, "count": 0}


def _over_daily_cap() -> bool:
    today = datetime.now(timezone.utc).date()
    if _daily["date"] != today:
        _daily["date"] = today
        _daily["count"] = 0
    if _daily["count"] >= settings.daily_request_cap:
        return True
    _daily["count"] += 1
    return False

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return StreamingResponse(
        iter([b'{"error":"rate_limit_exceeded"}']),
        status_code=429,
        media_type="application/json",
    )


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": settings.anthropic_model}


@app.post("/api/chat")
@limiter.limit(f"{settings.rate_limit_per_min}/minute")
async def chat(request: Request, body: ChatRequest):
    if _over_daily_cap():
        return StreamingResponse(
            iter([b'{"error":"daily_limit_reached"}']),
            status_code=429,
            media_type="application/json",
        )
    return StreamingResponse(
        answer_stream(body.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ingest")
async def ingest(x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="invalid admin key")
    from app.retrieval import reload_index
    from scripts.ingest import run_ingest

    count = run_ingest()
    reload_index()
    return {"status": "ok", "chunks_indexed": count}
