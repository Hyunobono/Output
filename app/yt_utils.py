from __future__ import annotations
import os, uuid, re, glob, tempfile
from typing import List, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp


# ---------------------------
# 유튜브 URL에서 video_id 추출
# ---------------------------
def extract_video_id(url: str) -> str:
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url


# ----------------------------------
# 1순위: 공식/자동 생성 자막 먼저 시도
# ----------------------------------
def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    vid = extract_video_id(url)
    try:
        # (1) 언어 우선순위 자막
        for code in (lang_priority or []):
            try:
                t = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                if t:
                    return t  # [{text, start, duration}, ...]
            except Exception:
                pass

        # (2) 자동 생성 자막 허용
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

        # (3) yt-dlp로 .vtt 다운로드 후 파싱
        t = fetch_captions_via_ytdlp(url, lang_priority)
        if t:
            return t  # [{start, end, text}, ...] 형태 (아래에서 텍스트만 쓰므로 OK)

        return None
    except TranscriptsDisabled:
        return None
    except Exception:
        return None


# -------------------------
# 오디오만 추출 (ffmpeg 필요)
# -------------------------
def download_audio(url: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, uuid.uuid4().hex)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base + ".%(ext)s",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}
        ],
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for ext in (".m4a", ".mp3", ".wav", ".aac", ".m4b"):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    raise RuntimeError("Audio download failed")


# --------------------------------------------------
# Whisper 로컬( faster-whisper )로 음성 → 자막 세그먼트
# --------------------------------------------------
def whisper_transcribe_local(audio_path: str, language_hint: Optional[str] = None) -> List[dict]:
    """
    Returns: List[dict] like:
      [{"text": "...", "start": 0.0, "end": 2.3}, ...]
    """
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "small")  # tiny/small/medium/large-v3 등
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_path,
        language=language_hint,
        vad_filter=True
    )
    out: List[dict] = []
    for s in segments:
        txt = (s.text or "").strip()
        if not txt:
            continue
        out.append({"text": txt, "start": float(s.start), "end": float(s.end)})
    return out


# -----------------------------------
# transcript → 순수 텍스트 (한 줄씩)
# -----------------------------------
def transcript_to_plaintext(trans: List[dict]) -> str:
    """
    YouTubeTranscriptApi/yt-dlp 파싱 결과의 공통 'text' 필드만 추출하여 줄 단위 연결
    """
    return "\n".join([x.get("text", "").strip() for x in trans if x.get("text")])


def whisper_to_plaintext(segments: List[dict]) -> str:
    """
    whisper_transcribe_local 결과를 줄 단위로 연결
    """
    return "\n".join([x["text"].strip() for x in segments if x.get("text")])


# ----------------------------------------------------
# yt-dlp로 .vtt 자막 받아서 파싱 (자동생성 포함 허용)
# ----------------------------------------------------
def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    """
    Returns list like: [{"start": 1.23, "end": 3.45, "text": "..."}]
    """
    from datetime import timedelta

    tmpdir = tempfile.mkdtemp(prefix="caps_")
    try:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,   # 자동 생성 자막도 허용
            "subtitlesformat": "vtt",
            "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
            "quiet": True,
            "noplaylist": True,
            # 우선순위 언어 + 보편 코드 후보
            "subtitleslangs": (lang_priority or []) + ["en", "en-US", "en-GB", "ko"]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid = info.get("id")

        # 다운로드된 vtt들
        vtts = glob.glob(f"{tmpdir}/{vid}*.vtt")
        if not vtts:
            return None

        # VTT 파서
        def parse_vtt(path: str) -> List[dict]:
            items: List[dict] = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip("\n") for l in f]

            ts_re = re.compile(r"(\d+:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d+:\d{2}:\d{2}\.\d{3})")
            buf_text: List[str] = []
            start_t = end_t = None

            def to_sec(ts: str) -> float:
                h, m, s_ms = ts.split(":")
                s, ms = s_ms.split(".")
                td = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
                return float(td.total_seconds())

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                m = ts_re.match(line)
                if m:
                    # 이전 청크 flush
                    if buf_text and start_t is not None:
                        txt = " ".join(buf_text).strip()
                        if txt:
                            items.append({"start": start_t, "end": end_t, "text": txt})
                        buf_text = []
                    start_t = to_sec(m.group(1))
                    end_t = to_sec(m.group(2))
                    i += 1
                    continue
                elif line == "" or line.startswith("WEBVTT"):
                    i += 1
                    continue
                else:
                    buf_text.append(line)
                    i += 1

            if buf_text and start_t is not None:
                txt = " ".join(buf_text).strip()
                if txt:
                    items.append({"start": start_t, "end": end_t, "text": txt})

            return items

        # 언어 우선 vtt 고르기
        prio = lang_priority or ["ko", "en"]
        for code in prio:
            cand = [p for p in vtts if f".{code}.vtt" in p]
            if cand:
                parsed = parse_vtt(cand[0])
                if parsed:
                    return parsed

        # 그래도 없으면 첫 번째 vtt 사용
        parsed = parse_vtt(vtts[0])
        return parsed if parsed else None

    except Exception:
        return None
    finally:
        # Render 같은 환경에서는 임시폴더 자동정리 안 될 수 있으니 굳이 삭제 X
        # shutil.rmtree(tmpdir, ignore_errors=True)  # 필요하면 주석 해제
        pass

