# app/yt_utils.py
from __future__ import annotations

import os
import re
import uuid
import glob
import tempfile
from typing import List, Optional

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


# ---------------------------
# 공통: 유저 에이전트 / 비디오ID
# ---------------------------
def _user_agent() -> str:
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

def extract_video_id(url: str) -> str:
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url


# -----------------------------------
# 1) youtube_transcript_api 로 우선 시도
# 2) 실패 시 yt-dlp 로 VTT 받아 파싱
# -----------------------------------
def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    vid = extract_video_id(url)
    prio = lang_priority or ["ko", "en"]

    # 1) 공식/자동 자막 API
    try:
        # 언어 우선
        for code in prio:
            try:
                t = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                if t:
                    return t  # [{text, start, duration}]
            except Exception:
                pass

        # 자동 생성 자막 허용
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(vid)
            for tr in transcripts:
                if tr.is_generated:
                    try:
                        t = tr.fetch()
                        if t:
                            return t
                    except Exception:
                        continue
        except Exception:
            pass
    except TranscriptsDisabled:
        pass
    except Exception:
        pass

    # 2) yt-dlp 로 .vtt 자막 내려받아 파싱
    return fetch_captions_via_ytdlp(url, prio)


# ------------------------------------------
# yt-dlp로 .vtt 자막(자동 포함) 받아서 파싱
# 반환: [{"start": float, "end": float, "text": str}, ...]
# ------------------------------------------
def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    cookies = os.getenv("YT_COOKIES_PATH")  # Secret Files 또는 ENV로 지정한 경로
    tmpdir = tempfile.mkdtemp(prefix="caps_")

    # 언어 후보 (우선순위 + 흔한 변형)
    langs: List[str] = []
    for c in (lang_priority or []):
        if c and c not in langs:
            langs.append(c)
    for extra in ["ko", "ko-KR", "en", "en-US", "en-GB"]:
        if extra not in langs:
            langs.append(extra)

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

        # 간단 VTT 파서
        from datetime import timedelta
        ts_re = re.compile(r"(\d+:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d+:\d{2}:\d{2}\.\d{3})")

        def to_sec(ts: str) -> float:
            h, m, s_ms = ts.split(":")
            s, ms = s_ms.split(".")
            td = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
            return float(td.total_seconds())

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
                        items.append({"start": start_t, "end": end_t if end_t else start_t + 2.0, "text": txt})
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

        # 언어 우선 매칭
        prio = lang_priority or ["ko", "en"]
        for code in prio:
            cand = [p for p in vtts if f".{code}.vtt" in p]
            if cand:
                parsed = parse_vtt(cand[0])
                if parsed:
                    return parsed

        # 아무거나 첫 파일
        parsed = parse_vtt(vtts[0])
        return parsed if parsed else None

    except Exception:
        return None
    finally:
        # 임시폴더 정리는 환경에 따라 생략 (Render는 짧은 수명 컨테이너)
        pass


# --------------------------
# 오디오 다운로드 (쿠키 지원)
# --------------------------
def download_audio(url: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, uuid.uuid4().hex)
    cookies = os.getenv("YT_COOKIES_PATH")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}],
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
        "http_headers": {"User-Agent": _user_agent()},
    }
    if cookies and os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for ext in (".m4a", ".mp3", ".wav", ".aac", ".m4b"):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    raise RuntimeError("Audio download failed")


# --------------------------------
# Whisper 로컬 STT (faster-whisper)
# --------------------------------
def whisper_transcribe_local(audio_path: str, language_hint: Optional[str] = None) -> List[dict]:
    """
    환경변수 WHISPER_MODEL (기본: tiny)
    반환: [{"text": str, "start": float, "end": float}, ...]
    """
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "tiny")  # tiny/small/medium/large-v3 등
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
# 텍스트 변환 유틸
# -----------------------
def transcript_to_plaintext(trans: List[dict]) -> str:
    return "\n".join([x.get("text", "").strip() for x in trans if x.get("text")])

def whisper_to_plaintext(segments: List[dict]) -> str:
    return "\n".join([x.get("text", "").strip() for x in segments if x.get("text")])

