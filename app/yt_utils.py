# app/yt_utils.py
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
# 공통: 유저 에이전트 / 비디오ID
# ---------------------------
def _user_agent() -> str:
    return (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


def extract_video_id(url: str) -> str:
    """유튜브 URL에서 video_id(11자) 추출. 못 찾으면 원문 반환."""
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url


# -----------------------------------
# 1) youtube_transcript_api 로 우선 시도
# 2) 실패 시 yt-dlp 로 VTT 받아 파싱
# -----------------------------------
def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    성공 시 [{"text": str, "start": float, "end": float}, ...] 반환. 실패 시 None.
    """
    vid = extract_video_id(url)
    prio = lang_priority or ["ko", "en"]

    # 1) 공식/자동 자막 API
    try:
        # 우선순위 언어 우선
        for code in prio:
            try:
                t = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                if t:
                    # API는 duration만 주므로 end를 만들어 맞춰줄 수도 있지만
                    # downstream에서 text만 쓰므로 그대로 반환
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
        # 자막 완전 비활성화
        pass
    except Exception:
        # 기타 에러는 조용히 넘어가고 yt-dlp 시도
        pass

    # 2) yt-dlp 로 .vtt 자막 내려받아 파싱
    return fetch_captions_via_ytdlp(url, prio)


# ------------------------------------------
# yt-dlp로 .vtt 자막(자동 포함) 받아서 파싱
# 반환: [{"start": float, "end": float, "text": str}, ...]
# ------------------------------------------
def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    YT_COOKIES_PATH 환경변수에 cookies.txt 경로가 있으면 인증 사용.
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
        "writeautomaticsub": True,           # 자동 생성 자막 허용
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

        # 다운로드된 vtt들
        vtts = glob.glob(f"{tmpdir}/{vid}*.vtt")
        if not vtts:
            return None

        # 간단 VTT 파서
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
                    # 타임스탬프 만나면 이전 블록 flush
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
        for code in langs:
            cand = [p for p in vtts if f".{code}." in p]
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
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------
# 오디오 다운로드 (쿠키 지원)
# --------------------------------
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

    # 확장자 탐색
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
    WHISPER_MODEL (기본 tiny) 사용.
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
    """자막 리스트(공식/자동/VTT파싱 공통) → 순수 텍스트"""
    return "\n".join([x.get("text", "").strip() for x in trans if x.get("text")])


def whisper_to_plaintext(segments: List[dict]) -> str:
    """Whisper 세그먼트 → 순수 텍스트"""
    return "\n".join([x.get("text", "").strip() for x in segments if x.get("text")])

