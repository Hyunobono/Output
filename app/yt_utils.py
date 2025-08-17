# app/yt_utils.py
from __future__ import annotations

import os
import re
import glob
import uuid
import tempfile
from typing import List, Optional
from datetime import timedelta

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


# ---------- 공통 유틸 ----------

def _user_agent() -> str:
    """간단한 브라우저 UA로 봇 차단 완화"""
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


def extract_video_id(url: str) -> str:
    """유튜브 URL에서 11자 video_id 추출. 실패 시 원문 반환."""
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url


# ---------- 자막(텍스트) 우선 수집 ----------

def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    1) youtube_transcript_api(공식/자동) 시도
    2) 실패 시 yt-dlp로 VTT 다운로드 후 파싱
    성공 시 [{"text": "...", "start": float, "end": float}] 또는
           youtube_transcript_api 원형({"text","start","duration"}) 반환
    실패 시 None
    """
    vid = extract_video_id(url)
    prio = lang_priority or ["ko", "en"]

    # 1) 공식/자동 자막 API
    try:
        # 언어 우선
        for code in prio:
            try:
                t = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                if t:
                    return t
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

    # 2) yt-dlp로 .vtt 자막 받아 파싱
    return fetch_captions_via_ytdlp(url, prio)


# ---------- yt-dlp로 VTT 받아서 파싱 ----------

def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    yt-dlp로 자막(.vtt) 내려받아 파싱.
    성공 시 [{"start": float, "end": float, "text": str}] 반환, 실패 시 None.
    환경변수 YT_COOKIES_PATH에 cookies.txt 경로가 있으면 인증 사용.
    """
    cookies = os.getenv("YT_COOKIES_PATH")
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

        # 타임스탬프 포맷 두 가지 지원: HH:MM:SS.mmm / MM:SS.mmm
        ts_hms = re.compile(r"(\d+):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2}\.\d{3})")
        ts_ms  = re.compile(r"(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d{2}):(\d{2}\.\d{3})")

        def to_sec_hms(h: str, m: str, s: str) -> float:
            return int(h) * 3600 + int(m) * 60 + float(s)

        def to_sec_ms(m: str, s: str) -> float:
            return int(m) * 60 + float(s)

        def parse_vtt(path: str) -> List[dict]:
            items: List[dict] = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.rstrip("\n") for ln in f]

            buf: List[str] = []
            start_t: Optional[float] = None
            end_t: Optional[float] = None

            def flush():
                nonlocal buf, start_t, end_t
                if buf and start_t is not None:
                    txt = " ".join(buf).strip()
                    if txt:
                        items.append({
                            "start": start_t,
                            "end": end_t if end_t is not None else start_t + 2.0,
                            "text": txt
                        })
                buf, start_t, end_t = [], None, None

            for line in lines:
                if not line or line.startswith("WEBVTT"):
                    continue
                m1 = ts_hms.match(line)
                m2 = ts_ms.match(line) if not m1 else None
                if m1:
                    flush()
                    sh, sm, ss, eh, em, es = m1.groups()
                    start_t = to_sec_hms(sh, sm, ss)
                    end_t   = to_sec_hms(eh, em, es)
                    continue
                if m2:
                    flush()
                    sm, ss, em, es = m2.groups()
                    start_t = to_sec_ms(sm, ss)
                    end_t   = to_sec_ms(em, es)
                    continue
                buf.append(line.strip())

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

        # 없으면 첫 번째 vtt
        parsed = parse_vtt(vtts[0])
        return parsed if parsed else None

    except Exception:
        return None
    finally:
        # 필요 시 임시폴더 정리:
        # import shutil; shutil.rmtree(tmpdir, ignore_errors=True)
        pass


# ---------- 오디오 다운로드 & Whisper ----------

def download_audio(url: str, outdir: str) -> str:
    """
    영상에서 오디오만 추출(m4a/…)
    YT_COOKIES_PATH 있으면 쿠키 사용.
    """
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, uuid.uuid4().hex)
    cookies = os.getenv("YT_COOKIES_PATH")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base + ".%(ext)s",
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
    WHISPER_MODEL (기본 tiny) 사용.
    반환: [{"text": str, "start": float, "end": float}, ...]
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


# ---------- 텍스트 변환 ----------

def transcript_to_plaintext(trans: List[dict]) -> str:
    """자막 리스트(공식/자동/VTT파싱 공통) → 순수 텍스트"""
    return "\n".join([x.get("text", "").strip() for x in trans if x.get("text")])


def whisper_to_plaintext(segments: List[dict]) -> str:
    """Whisper 세그먼트 → 순수 텍스트"""
    return "\n".join([x.get("text", "").strip() for x in segments if x.get("text")])

