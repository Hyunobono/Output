from __future__ import annotations
import os, tempfile, shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.yt_utils import (
    fetch_transcript_if_available,
    download_audio,
    whisper_transcribe_local,
    list_of_dicts_to_plaintext,   # ✅ 여기로 변경
)
from app.summarizer import summarize_long_text

app = FastAPI(title="YT Auto Summarizer", version="1.0.0")

class SummarizeReq(BaseModel):
    url: str
    lang_priority: list[str] | None = ["ko", "en"]
    whisper_lang_hint: str | None = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/summarize")
def summarize(req: SummarizeReq):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise HTTPException(500, detail="Missing GEMINI_API_KEY")

    tmpdir = tempfile.mkdtemp(prefix="yt_auto_")
    try:
        # 1) 자막 우선
        trans = fetch_transcript_if_available(req.url, req.lang_priority or ["ko", "en"])
        method = "captions"
        if trans:
            text = list_of_dicts_to_plaintext(trans)   # ✅ 여기로 변경
        else:
            # 2) 자막 없으면 Whisper 로컬
            method = "whisper"
            audio = download_audio(req.url, tmpdir)
            segs = whisper_transcribe_local(audio, language_hint=req.whisper_lang_hint)
            text = list_of_dicts_to_plaintext(segs)    # ✅ 여기로 변경

        if not text or len(text.strip()) < 20:
            raise HTTPException(400, detail="Transcript empty or too short")

        result = summarize_long_text(GEMINI_API_KEY, text)
        return {
            "source_url": req.url,
            "method": method,
            "summary": result["final"],
            "partials": result["partials"],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

