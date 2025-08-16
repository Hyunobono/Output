import os
import re
import glob
import uuid
import tempfile
from typing import List, Optional

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


# -----------------------------
# 자막(공식/자동) 우선 시도
# -----------------------------
def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    YouTubeTranscriptApi로 영상 자막(우선순위 언어 → 자동생성)을 순서대로 시도.
    성공 시 [{"text": "...", "start": 0.0, "end": 2.3}, ...] 형태 반환.
    실패 시 None.
    """
    # video_id 추출 (대부분 11자)
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    vid = m.group(1) if m else url

    try:
        # 1) 우선순위 언어 자막
        for code in (lang_priority or []):
            try:
                t = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                if t:
                    return t
            except Exception:
                pass

        # 2) 자동 생성 자막 허용
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
        for tr in transcripts:
            if tr.is_generated:
                try:
                    t = tr.fetch()
                    if t:
                        return t
                except Exception:
                    continue

        return None
    except TranscriptsDisabled:
        return None
    except Exception:
        return None


# ------------------------------------------
# yt-dlp로 .vtt 자막(자동 포함) 받아서 파싱
# ------------------------------------------
def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    yt-dlp로 .vtt 자막을 내려받아 파싱.
    성공 시 [{"start": float, "end": float, "text": str}, ...] 반환. 실패 시 None.
    YT_COOKIES_PATH 환경변수에 cookies.txt 경로가 있으면 인증 사용.
    """
    from datetime import timedelta

    cookies = os.getenv("YT_COOKIES_PATH")
    tmpdir = tempfile.mkdtemp(prefix="caps_")
    try:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,  # 자동생성 허용
            "subtitlesformat": "vtt",
            "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
            "quiet": True,
            "noplaylist": True,
            "subtitleslangs": (lang_priority or []) + ["en", "en-US", "en-GB", "ko"],
        }
        if cookies and os.path.exists(cookies):
            ydl_opts["cookiefile"] = cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid = info.get("id")

        vtts = glob.glob(f"{tmpdir}/{vid}*.vtt")
        if not vtts:
            return None

        # 간단 VTT 파서
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
                        items.append({"start": start_t, "end": end_t if end_t else start_t, "text": txt})
                buf_text, start_t, end_t = [], None, None

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                m = ts_re.match(line)
                if m:
                    flush()
                    start_t = to_sec(m.group(1))
                    end_t = to_sec(m.group(2))
                    i += 1
                    continue
                if not line or line.startswith("WEBVTT"):
                    i += 1
                    continue
                buf_text.append(line)
                i += 1

            flush()
            return items

        # 언어 우선
        prio = lang_priority or ["ko", "en"]
        for code in prio:
            cand = [p for p in vtts if f".{code}.vtt" in p]
            if cand:
                parsed = parse_vtt(cand[0])
                if parsed:
                    return parsed

        # 없다면 첫 번째 vtt
        parsed = parse_vtt(vtts[0])
        return parsed if parsed else None

    except Exception:
        return None


# --------------------------------
# 오디오 다운로드 (쿠키 지원)
# --------------------------------
def download_audio(url: str, outdir: str) -> str:
    """
    영상에서 오디오 추출(m4a/…)
    YT_COOKIES_PATH 있으면 인증 쿠키 사용.
    """
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, uuid.uuid4().hex)

    cookies = os.getenv("YT_COOKIES_PATH")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}],
        "quiet": True,
        "noplaylist": True,
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

    model_name = os.getenv("WHISPER_MODEL", "tiny")  # tiny/small/medium/large-v3 …
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

