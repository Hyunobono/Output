import os
import re
import json
import time
import requests
import subprocess
import tempfile
import shutil
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

# ---------- 유틸 ----------
def extract_video_id(youtube_url: str) -> str:
    u = urlparse(youtube_url)
    if u.netloc in ("youtu.be",):
        return u.path.strip("/")
    if u.netloc.endswith("youtube.com"):
        qs = parse_qs(u.query)
        if "v" in qs: return qs["v"][0]
    m = re.search(r"(?:v=|/embed/|youtu\.be/)([\w-]{11})", youtube_url)
    return m.group(1) if m else None

def chunk_text(s: str, max_chars=8000):
    chunks, buf, size = [], [], 0
    for line in s.splitlines():
        line = line.strip()
        if not line: continue
        if size + len(line) + 1 > max_chars:
            chunks.append("\n".join(buf)); buf=[]; size=0
        buf.append(line); size += len(line) + 1
    if buf: chunks.append("\n".join(buf))
    return chunks

# ---------- 자막 → 실패 시 Whisper ----------
def fetch_transcript_text(video_id: str):
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko","en"])
        return " ".join(x["text"] for x in srt if x["text"].strip())
    except Exception:
        return None

def download_audio(video_url: str, out_path: str):
    subprocess.run(["yt-dlp","-x","--audio-format","m4a","-o",out_path, video_url], check=True)

def transcribe_whisper(audio_path: str, model_size="base"):
    import whisper
    model = whisper.load_model(model_size)
    r = model.transcribe(audio_path, temperature=0)
    return r["text"].strip()

# ---------- Gemini 호출 공통 ----------
def _gemini(payload):
    api_key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    for attempt in range(5):
        r = requests.post(url, json=payload, timeout=180)
        if r.status_code in (500,502,503,504,429):
            time.sleep(1.5**attempt); continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()

def _extract_text(j):
    try:
        return j["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(j, ensure_ascii=False)

# ---------- 요약/확장/솔루션 ----------
def summarize_brief(text: str) -> str:
    prompt = (
        "너는 한국어 요약가다. 아래 텍스트를 **5~7줄 bullet**로 간결하게 요약하라. "
        "마지막에 '**핵심 주장**'과 '**근거**' 섹션을 분리해라.\n\n=== 원문 ===\n" + text[:18000]
    )
    j = _gemini({"contents":[{"parts":[{"text":prompt}]}]})
    return _extract_text(j)

def summarize_detailed(text: str) -> str:
    prompt = (
        "너는 한국어 분석가다. 아래 텍스트의 **핵심 내용**을 더 길고 구체적으로 정리하라. "
        "구성은 다음 형식을 반드시 지켜라:\n"
        "1) 핵심 요지(3~5문장)\n"
        "2) 주요 포인트 8~12개(불릿, 각 1~2문장)\n"
        "3) 논쟁 지점/오해 가능성 3~5개\n"
        "4) 데이터/숫자/사례가 있다면 발췌 정리(없으면 생략)\n"
        "5) 핵심 주장 vs 근거(각 3~5개 불릿)\n"
        "6) 요약 결론(2~4문장)\n\n=== 원문 ===\n" + text[:18000]
    )
    j = _gemini({"contents":[{"parts":[{"text":prompt}]}]})
    return _extract_text(j)

def summarize_large(text: str) -> str:
    # 텍스트가 길면 부분 요약→최종 통합
    parts = []
    chunks = chunk_text(text, 8000)
    for i, ch in enumerate(chunks, 1):
        j = _gemini({"contents":[{"parts":[{"text":f"[부분 {i}/{len(chunks)}] 다음을 간결히 요약:\n{ch}"}]}]})
        parts.append(f"[부분요약 {i}]\n{_extract_text(j)}")
    final_prompt = (
        "아래 부분요약들을 **길고 구체적인 통합 요약**으로 재구성하라. "
        "위의 'summarize_detailed' 형식을 따르되, 중복을 줄이고 구조를 명확히 하라.\n\n" + "\n\n".join(parts)
    )
    j = _gemini({"contents":[{"parts":[{"text":final_prompt}]}]})
    return _extract_text(j)

def propose_solutions(summary_or_text: str) -> str:
    prompt = (
        "너는 실전 컨설턴트다. 아래 내용을 기반으로 **실행 솔루션 7개**를 제시하라. "
        "각 항목은 ①짧은 제목 ②왜 필요한지(1문장) ③3단계 체크리스트 로 구성하고, "
        "가능하면 수치적 기준·마감기한·도구 예시를 포함하라. 한국어로 작성하라.\n\n=== 내용 ===\n"
        + summary_or_text[:18000]
    )
    j = _gemini({"contents":[{"parts":[{"text":prompt}]}]})
    return _extract_text(j)

def extras(summary_or_text: str) -> str:
    prompt = (
        "아래 내용을 더 깊게 이해하기 위한 **덤 정보**를 제공하라.\n"
        "- 관련 핵심 용어 6~10개(간단 정의)\n"
        "- 추가로 보면 좋은 관점 3~5개\n"
        "- 추천 검색 쿼리 6~10개(한국어/영어 혼합)\n"
        "- 주의할 함정 또는 반례 3~5개\n\n=== 내용 ===\n" + summary_or_text[:18000]
    )
    j = _gemini({"contents":[{"parts":[{"text":prompt}]}]})
    return _extract_text(j)

# ---------- 메인 파이프라인 ----------
def run_pipeline(youtube_url: str, whisper_model="base"):
    vid = extract_video_id(youtube_url)
    if not vid:
        raise ValueError("유효한 YouTube 링크가 아님")

    # 1) 자막 먼저
    text = fetch_transcript_text(vid)

    # 2) 없으면 Whisper
    if not text or not text.strip():
        tmp = tempfile.mkdtemp(prefix="yt_")
        try:
            audio = os.path.join(tmp, "audio.m4a")
            download_audio(youtube_url, audio)
            text = transcribe_whisper(audio, model_size=whisper_model)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # 3) 요약(짧은+긴)
    brief = summarize_brief(text)
    detailed = summarize_large(text) if len(text) > 10000 else summarize_detailed(text)

    # 4) 솔루션 + 덤
    sols = propose_solutions(detailed or brief)
    extra = extras(detailed or brief)

    return {
        "video_id": vid,
        "summary_brief": brief,
        "summary_detailed": detailed,
        "solutions": sols,
        "extras": extra
    }

# ---------- 직접 실행 ----------
if __name__ == "__main__":
    url = input("YouTube URL: ").strip()
    out = run_pipeline(url)
    print(json.dumps(out, ensure_ascii=False, indent=2))
