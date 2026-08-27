# -*- coding: utf-8 -*-
"""
Off The Record -- scratch analysis API.

    POST /analyze-record
                     four photographs, two of each side -> all four marked, a
                     grade per side, and a grade for the record. This is the
                     endpoint the app uses.
    POST /analyze    one side on its own: an image -> detected marks, coverage,
                     suggested grade, and the image with the marks highlighted.
                     Send an optional second shot of the same side and each mark
                     is also checked against it.
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
from scanner import analyze_record as run_record_analysis

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


def _check_upload(data, what):
    if not data:
        raise HTTPException(400, f"empty upload: {what}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"{what} is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    if _sniff(data) is None:
        raise HTTPException(
            415, f"unsupported file type for {what}; send a JPEG, PNG, WEBP or BMP")


@app.post("/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    second_image: UploadFile = File(
        None, description="another shot of the SAME side, lamp moved"),
    overlay: bool = Query(True, description="include the highlighted image"),
):
    """Grade one side of a record.

    Sending a second shot of the same side — same disc, lamp moved between the
    two — lets every mark be checked against it. Marks that show up in both are
    far more likely to be real damage rather than a reflection, and they weigh
    more in the grade accordingly. The two photos do not need to be lined up by
    the caller; the rotation is worked out from the centre label.
    """
    data = await image.read()
    _check_upload(data, "image")

    second = None
    if second_image is not None:
        second = await second_image.read()
        if second:
            _check_upload(second, "second_image")
        else:
            second = None

    try:
        result = run_analysis(data, want_overlay=overlay, second_image_bytes=second)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["analysis_id"] = uuid.uuid4().hex
    return result


@app.post("/analyze-record")
async def analyze_record_endpoint(
    side_a_1: UploadFile = File(..., description="side A, first tilt"),
    side_a_2: UploadFile = File(..., description="side A, second tilt"),
    side_b_1: UploadFile = File(..., description="side B, first tilt"),
    side_b_2: UploadFile = File(..., description="side B, second tilt"),
    overlay: bool = Query(True, description="include the marked photographs"),
):
    """Grade a whole record from four photographs — two of each side.

    The two shots of a side must be of the SAME side with the disc tilted
    differently between them, so the lamp rakes across the surface from a
    different angle. They do not need to be lined up by the caller: the rotation
    between them is read off the centre label. Each photograph must show the
    WHOLE disc, label included, or the disc cannot be located.

    Every photograph comes back marked. The record's grade is the worse of its
    two sides, never their average — a buyer plays both.
    """
    files = {"A": [side_a_1, side_a_2], "B": [side_b_1, side_b_2]}
    sides = {}
    for name, uploads in files.items():
        shots = []
        for i, up in enumerate(uploads, 1):
            data = await up.read()
            _check_upload(data, f"side {name} photo {i}")
            shots.append(data)
        sides[name] = shots

    try:
        result = run_record_analysis(sides, want_overlay=overlay)
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
