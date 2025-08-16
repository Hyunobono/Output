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


# ---------- 공통 유틸 ----------

def extract_video_id(url: str) -> str:
    """
    유튜브 URL에서 video_id(11자) 추출. 못 찾으면 원문 반환.
    """
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url


def _user_agent() -> str:
    # 봇 차단 완화용 간단 UA
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


# ---------- 자막(텍스트) 우선 수집 ----------

def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    1) youtube_transcript_api (공식/자동 자막)
    2) yt-dlp 로 .vtt 받아 파싱 (자동 자막 허용)
    둘 다 실패 시 None
    """
    vid = extract_video_id(url)
    prio = lang_priority or ["ko", "en"]

    # 1) 공식/자동 자막 API 시도
    try:
        # 언어 우선순위로 공식 자막 먼저
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

    # 2) yt-dlp 로 .vtt 내려받아 파싱
    return fetch_captions_via_ytdlp(url, prio)


def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    yt-dlp 로 자막(.vtt) 내려받아 transcript 형태로 파싱.
    자동 생성 자막 허용.
    반환 형식: [{"start": float, "end": float, "text": str}, ...]
    """
    cookies = os.getenv("YT_COOKIES_PATH")  # Secret Files로 올린 경로(/etc/secrets/cookies.txt 등)
    tmpdir = tempfile.mkdtemp(prefix="caps_")

    # 언어 후보 (우선순위 + 흔한 변형)
    langs = []
    for c in (lang_priority or []):
        if c and c not in langs:
            langs.append(c)
    for extra in ["ko", "ko-KR", "en", "en-US", "en-GB"]:
        if extra not in langs:
            langs.append(extra)

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,              # 자동 생성 자막 허용
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

        # 내려받은 vtt 후보들
        vtts = glob.glob(f"{tmpdir}/{vid}*.vtt")
        if not vtts:
            return None

        # 언어 우선순위에 맞는 vtt 먼저 찾기
        for code in langs:
            cand = [p for p in vtts if f".{code}.vtt" in p]
            if cand:
                parsed = _parse_vtt_to_segments(cand[0])
                if parsed:
                    return parsed

        # 그래도 없으면 첫 번째 vtt 사용
        parsed = _parse_vtt_to_segments(vtts[0])
        return parsed if parsed else None

    except Exception:
        return None
    finally:
        # 임시 디렉토리는 호출부에서 지우고 싶다면 지워도 됨
        pass


def _parse_vtt_to_segments(path: str) -> List[dict]:
    """
    단순 VTT 파서 → [{"start": float, "end": float, "text": str}, ...]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    ts_re = re.compile(r"(\d+):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2}\.\d{3})")
    items: List[dict] = []
    buf: List[str] = []
    start_t: Optional[float] = None
    end_t: Optional[float] = None

    def flush():
        nonlocal buf, start_t, end_t
        if buf and start_t is not None:
            txt = " ".join(buf).strip()
            if txt:
                items.append({"start": start_t, "end": end_t if end_t else start_t + 2.0, "text": txt})
        buf, start_t, end_t = [], None, None

    def to_sec(h: str, m: str, s: str) -> float:
        hh = int(h)
        mm = int(m)
        ss = float(s)
        return hh * 3600 + mm * 60 + ss

    for line in lines:
        if not line or line.startswith("WEBVTT"):
            continue
        m = ts_re.search(line)
        if m:
            flush()
            sh, sm, ss, eh, em, es = m.groups()
            start_t = to_sec(sh, sm, ss)
            end_t = to_sec(eh, em, es)
            continue
        # 본문
        buf.append(line.strip())

    flush()
    return items


# ---------- 오디오 다운로드 & Whisper ----------

def download_audio(url: str, outdir: str) -> str:
    """
    유튜브에서 오디오만 추출(m4a/…)
    - 쿠키 있으면 bot 차단 회피에 도움
    """
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, uuid.uuid4().hex)
    cookies = os.getenv("YT_COOKIES_PATH")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base + ".%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
        "http_headers": {"User-Agent": _user_agent()},
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}
        ],
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


def whisper_transcribe_local(audio_path: str, language_hint: Optional[str] = None) -> List[dict]:
    """
    faster-whisper 로 로컬 STT.
    환경변수 WHISPER_MODEL (기본: tiny)
    반환: [{"text", "start", "end"}, ...]
    """
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "tiny")  # tiny/small/medium/large-v3 등
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language=language_hint, vad_filter=True)

    out: List[dict] = []
    for s in segments:
        out.append({"text": (s.text or "").strip(), "start": float(s.start or 0.0), "end": float(s.end or 0.0)})
    return out


# ---------- 텍스트 변환 ----------

def transcript_to_plaintext(trans: List[dict]) -> str:
    """
    youtube_transcript_api 또는 VTT 파싱 결과를 한 덩어리 텍스트로
    """
    return "\n".join([x.get("text", "").strip() for x in trans if x.get("text")])


def whisper_to_plaintext(segments: List[dict]) -> str:
    """
    whisper_transcribe_local 결과를 한 덩어리 텍스트로
    """
    return "\n".join([x.get("text", "").strip() for x in segments if x.get("text")])

