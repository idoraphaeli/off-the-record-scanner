# -*- coding: utf-8 -*-
"""
Off The Record -- scratch analysis API.

    POST /analyze    an image -> detected marks, coverage, suggested grade,
                     and an overlay image with the marks highlighted
    POST /feedback   an admin's verdict on one detection (was it a real scratch?)
    GET  /feedback   list what has been collected so far (admin)
    GET  /health     readiness probe

Auth: the analyze endpoint is open; the feedback endpoints require the header
`X-Admin-Token` to match the ADMIN_TOKEN environment variable. Feedback is the
one thing that can corrupt the training data, so it is the one thing gated.
"""

import json
import os
import time
import uuid

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from scanner import analyze as run_analysis

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
# JPEG, PNG, WEBP, BMP. Checked from the bytes themselves -- a file extension or
# a client-supplied content-type is caller input and can say anything.
MAGIC = ((b"\xff\xd8\xff", "jpeg"), (b"\x89PNG\r\n\x1a\n", "png"),
         (b"RIFF", "webp"), (b"BM", "bmp"))

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "/tmp/feedback.jsonl")
ALLOWED_ORIGINS = [o.strip() for o in
                   os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Off The Record scanner", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _sniff(data: bytes):
    for magic, kind in MAGIC:
        if data.startswith(magic):
            return kind
    return None


def _require_admin(token):
    if not ADMIN_TOKEN:
        raise HTTPException(503, "feedback is disabled: ADMIN_TOKEN is not set")
    if token != ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")


@app.get("/health")
def health():
    return {"status": "ok", "feedback_enabled": bool(ADMIN_TOKEN)}


@app.get("/capture")
def capture_page():
    """Dataset capture tool: a phone page that enforces the same framing circle
    and level check as the app, but downloads the photo to the device instead of
    uploading it. Served from here because a live camera needs HTTPS, which a
    local file cannot provide."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "capture.html"),
                        media_type="text/html")


@app.post("/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    overlay: bool = Query(True, description="include the highlighted image"),
):
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    if _sniff(data) is None:
        raise HTTPException(415, "unsupported file type; send a JPEG, PNG, WEBP or BMP")

    try:
        result = run_analysis(data, want_overlay=overlay)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["analysis_id"] = uuid.uuid4().hex
    return result


class Feedback(BaseModel):
    analysis_id: str = Field(..., description="from the /analyze response")
    mark_index: int = Field(..., ge=0, description="which mark in `marks`")
    is_scratch: bool = Field(..., description="admin verdict on this mark")
    note: str = Field("", max_length=500)
    record_id: str = Field("", max_length=120)


@app.post("/feedback")
def submit_feedback(item: Feedback, x_admin_token: str = Header(default="")):
    _require_admin(x_admin_token)
    row = item.model_dump()
    row["ts"] = time.time()
    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"stored": True}


@app.get("/feedback")
def list_feedback(x_admin_token: str = Header(default=""), limit: int = 500):
    _require_admin(x_admin_token)
    if not os.path.exists(FEEDBACK_PATH):
        return {"count": 0, "items": []}
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {"count": len(rows), "items": rows[-limit:]}
