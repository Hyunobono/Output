import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.yt_utils import (
    download_audio,
    fetch_transcript_if_available,
    list_of_dicts_to_plaintext,
    whisper_transcribe_local,
)


# ===================================================================
# FastAPI App
# ===================================================================
app = FastAPI(
    title="AI-powered YouTube Video Summarizer",
    description="Summarize a YouTube video and analyze its content.",
    version="1.0.0",
)


@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    return {"message": "Service is healthy"}


@app.get("/.well-known/openapi.yaml", include_in_schema=False)
async def get_openapi_spec():
    return FileResponse("openapi.yaml", media_type="text/yaml")


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def get_ai_plugin():
    return {
        "schema_version": "v1",
        "name_for_model": "YouTube_Summarizer",
        "name_for_human": "YouTube Summarizer",
        "description_for_model": "Tool for summarizing YouTube videos given a video URL.",
        "description_for_human": "A tool that summarizes YouTube videos.",
        "auth": {"type": "none"},
        "api": {"type": "openapi", "url": os.getenv("RENDER_EXTERNAL_URL") + "/.well-known/openapi.yaml"},
        "logo_url": "https://output-xrhd.onrender.com/logo.png",
        "contact_email": "support@yourdomain.com",
    }


class SummarizeRequest(BaseModel):
    url: str
    lang_priority: Optional[List[str]] = ["ko", "en"]
    whisper_lang_hint: Optional[str] = None


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    tmpdir = None
    try:
        # Step 1: Try to fetch official/auto-generated transcripts first
        transcript = fetch_transcript_if_available(req.url, req.lang_priority)
        if transcript:
            summary_method = "captions"
            plain_text = list_of_dicts_to_plaintext(transcript)
            # Placeholder for actual analysis/summary
            result = {
                "source_url": req.url,
                "method": summary_method,
                "summary": plain_text,  # Use plaintext as summary for now
            }
            return result

        # Step 2: If no transcript, download audio and use Whisper
        tmpdir = tempfile.mkdtemp()
        audio = download_audio(req.url, tmpdir)
        transcript_whisper = whisper_transcribe_local(audio, req.whisper_lang_hint)

        summary_method = "whisper"
        plain_text = list_of_dicts_to_plaintext(transcript_whisper)
        # Placeholder for actual analysis/summary
        result = {
            "source_url": req.url,
            "method": summary_method,
            "summary": plain_text,  # Use plaintext as summary for now
        }
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {e}",
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir)
