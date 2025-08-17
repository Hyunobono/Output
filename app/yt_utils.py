from typing import List, Optional
import os
import re
import glob
import uuid
import tempfile
import shutil

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


# ---------------------------
# Common: User Agent / Video ID
# ---------------------------
def _user_agent() -> str:
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


def extract_video_id(url: str) -> str:
    """Extracts the 11-character video_id from a YouTube URL. Returns the original string if not found."""
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url


# -----------------------------------
# 1) Try with youtube_transcript_api first
# 2) Fallback to yt-dlp to download and parse VTT
# -----------------------------------
def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    Returns [{"text": str, "start": float, "end": float}, ...] or
    the original youtube_transcript_api format ({"text", "start", "duration"}) on success. Returns None on failure.
    """
    vid = extract_video_id(url)
    prio = lang_priority or ["ko", "en"]

    # 1) Official/auto-generated subtitle API
    try:
        # Try prioritized languages
        for code in prio:
            try:
                t = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                if t:
                    return t
            except Exception:
                pass

        # Allow for auto-generated subtitles
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
        for tr in transcripts:
            if tr.is_generated:
                try:
                    t = tr.fetch()
                    if t:
                        return t
                except Exception:
                    continue

    except TranscriptsDisabled:
        pass
    except Exception:
        pass

    # 2) Fallback: Use yt-dlp to download and parse .vtt subtitles
    return fetch_captions_via_ytdlp(url, prio)


# ------------------------------------------
# Use yt-dlp to download and parse .vtt subtitles (including auto-generated)
# Returns: [{"start": float, "end": float, "text": str}, ...]
# ------------------------------------------
def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    Uses authentication if YT_COOKIES_PATH environment variable is set.
    """
    cookies = os.getenv("YT_COOKIES_PATH")
    tmpdir = tempfile.mkdtemp(prefix="caps_")

    # Language candidates (priority + common variants)
    langs = list(dict.fromkeys(lang_priority + ["ko", "ko-KR", "en", "en-US", "en-GB"]))

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
        "http_headers": {"User-Agent": _user_agent()},
    }
    if cookies and os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid = info.get("id")

        vtts = glob.glob(f"{tmpdir}/{vid}*.vtt")
        if not vtts:
            return None

        # Timestamp regex (HH:MM:SS.mmm --> HH:MM:SS.mmm)
        ts_re = re.compile(r"(\d+:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d+:\d{2}:\d{2}\.\d{3})")

        def to_sec(ts: str) -> float:
            h, m, s_ms = ts.split(":")
            s, ms = s_ms.split(".")
            return int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000.0

        def parse_vtt(path: str) -> List[dict]:
            items: List[dict] = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.rstrip("\n") for l in f]

            buf_text: List[str] = []
            start_t: Optional[float] = None
            end_t: Optional[float] = None

            def flush():
                nonlocal buf_text, start_t, end_t
                if buf_text and start_t is not None:
                    txt = " ".join(buf_text).strip()
                    if txt:
                        items.append({
                            "start": start_t,
                            "end": end_t if end_t is not None else (start_t + 2.0),
                            "text": txt
                        })
                buf_text, start_t, end_t = [], None, None

            for line in lines:
                line_s = line.strip()
                m = ts_re.match(line_s)
                if m:
                    flush()
                    start_t = to_sec(m.group(1))
                    end_t = to_sec(m.group(2))
                    continue
                if not line_s or line_s.startswith("WEBVTT"):
                    continue
                buf_text.append(line_s)

            flush()
            return items

        for code in langs:
            cand = [p for p in vtts if f".{code}." in p]
            if cand:
                parsed = parse_vtt(cand[0])
                if parsed:
                    return parsed

        parsed = parse_vtt(vtts[0])
        return parsed if parsed else None

    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------
# Download audio (with cookies support)
# --------------------------------
def download_audio(url: str, outdir: str) -> str:
    """
    Extracts only the audio from a video (m4a/...).
    Uses cookies if YT_COOKIES_PATH is set.
    """
    os.makedirs(outdir, exist_ok=True)
    base_name = uuid.uuid4().hex
    cookies = os.getenv("YT_COOKIES_PATH")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(outdir, f"{base_name}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}
        ],
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
        "http_headers": {"User-Agent": _user_agent()},
    }
    if cookies and os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if filename and os.path.exists(filename):
                return filename

    except Exception as e:
        raise RuntimeError(f"Audio download failed: {e}")

    raise RuntimeError("Audio download failed to return a file path.")


# --------------------------------
# Local Whisper STT (faster-whisper)
# --------------------------------
def whisper_transcribe_local(audio_path: str, language_hint: Optional[str] = None) -> List[dict]:
    """
    Uses WHISPER_MODEL (default: tiny).
    Returns: [{"text": str, "start": float, "end": float}, ...]
    """
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "tiny")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(audio_path, language=language_hint, vad_filter=True)

    out: List[dict] = []
    for s in segments:
        txt = (s.text or "").strip()
        if not txt:
            continue
        out.append({"text": txt, "start": float(s.start or 0.0), "end": float(s.end or 0.0)})
    return out


# -----------------------
# Text conversion utility
# -----------------------
def list_of_dicts_to_plaintext(data: List[dict]) -> str:
    """Converts a list of dictionaries with a 'text' key into a single plaintext string."""
    return "\n".join([x.get("text", "").strip() for x in data if x.get("text")])
