import os, uuid, re, glob, tempfile
from typing import List, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import yt_dlp

# 유튜브 URL에서 video_id 추출
def extract_video_id(url: str) -> str:
    m = re.search(r'(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else url

# 유튜브 자막 가져오기
def fetch_transcript_if_available(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    vid = extract_video_id(url)
    try:
        # 1) 언어 우선순위 자막 가져오기
        for code in lang_priority:
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
        # 3) yt-dlp 기반 자막
        t = fetch_captions_via_ytdlp(url, lang_priority)
        if t:
            return t
        return None
    except TranscriptsDisabled:
        return None
    except Exception:
        return None

# 오디오 다운로드
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
    for ext in (".m4a",".mp3",".wav",".aac",".m4b"):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    raise RuntimeError("Audio download failed")

# Whisper 로컬 실행
def whisper_transcribe_local(audio_path: str, language_hint: Optional[str] = None) -> List[dict]:
    from faster_whisper import WhisperModel
    model_name = os.getenv("WHISPER_MODEL", "small")  # small / medium / large-v3
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language=language_hint, vad_filter=True)
    return [{"text": s.text.strip(), "start": s.start, "end": s.end} for s in segments]

# transcript -> 텍스트 변환
def transcript_to_plaintext(trans: List[dict]) -> str:
    return "\n".join([x.get("text","").strip() for x in trans if x.get("text")])

def whisper_to_plaintext(segments: List[dict]) -> str:
    return "\n".join([x["text"] for x in segments if x.get("text")])

# yt-dlp 자막 다운로드 후 파싱
def fetch_captions_via_ytdlp(url: str, lang_priority: List[str]) -> Optional[List[dict]]:
    from datetime import timedelta
    tmpdir = tempfile.mkdtemp(prefix="caps_")
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": lang_priority + ["en", "en-US", "en-GB", "ko"],
        "subtitlesformat": "vtt",
        "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        files = glob.glob(f"{tmpdir}/*.vtt")
        if not files:
            return None

        path = files[0]
        lines = open(path, encoding="utf-8").read().splitlines()
        results = []
        text_accum = []
        start_time = end_time = None

        def flush():
            nonlocal text_accum, start_time, end_time
            if text_accum and start_time is not None:
                results.append({
                    "start": start_time,
                    "end": end_time if end_time else start_time + 2,
                    "text": " ".join(text_accum).strip()
                })
            text_accum, start_time, end_time = [], None, None

        ts_pattern = re.compile(r"(\d+):(\d+):(\d+\.\d+) --> (\d+):(\d+):(\d+\.\d+)")
        for line in lines:
            m = ts_pattern.search(line)
            if m:
                flush()
                sh, sm, ss, eh, em, es = m.groups()
                start_time = int(sh) * 3600 + int(sm) * 60 + float(ss)
                end_time = int(eh) * 3600 + int(em) * 60 + float(es)
                continue
            if line.strip() and not line.startswith("WEBVTT"):
                text_accum.append(line.strip())
        flush()
        return results if results else None
    except Exception:
        return None

