# server.py (FastAPI 버전)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal

from transcript_pipeline import (
    fetch_transcript_if_available,
    transcript_to_plaintext,
    download_audio,
    whisper_transcribe_local,
    whisper_to_plaintext,
)
from summarizer import summarize_long_text

app = FastAPI(title="YT Auto Summarizer", version="1.0.0")

class SummReq(BaseModel):
    url: HttpUrl
    lang: str = Field(default="ko", description="ko/en 등")
    provider: Literal["gemini", "openai"] = Field(default="gemini")
    force_whisper: bool = Field(default=False)

class SummResp(BaseModel):
    ok: bool
    summary: str
    meta: dict

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/summarize", response_model=SummResp)
def summarize(req: SummReq):
    try:
        # 1) 자막 우선 → 없으면 Whisper
        text = None
        used = {"pipeline": None}
        if not req.force_whisper:
            trans = fetch_transcript_if_available(str(req.url), [req.lang, "en"])
            if trans:
                text = transcript_to_plaintext(trans)
                used["pipeline"] = "captions"
        if not text:
            audio = download_audio(str(req.url), "out")
            segs = whisper_transcribe_local(audio, language_hint=req.lang)
            text = whisper_to_plaintext(segs)
            used["pipeline"] = "whisper"

        # 2) 요약/분석
        out = summarize_long_text(text, provider=req.provider, lang=req.lang)

        return SummResp(
            ok=True,
            summary=out,
            meta={"lang": req.lang, "provider": req.provider, "path": used["pipeline"], "text_len": len(text)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

