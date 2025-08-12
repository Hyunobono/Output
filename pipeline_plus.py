import os
import re
import json
import requests
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()  # .env에서 환경변수 불러오기

def extract_video_id(youtube_url: str) -> str:
    u = urlparse(youtube_url)
    if u.netloc in ("youtu.be",):
        return u.path.strip("/")
    if u.netloc.endswith("youtube.com"):
        qs = parse_qs(u.query)
        if "v" in qs:
            return qs["v"][0]
    m = re.search(r"(?:v=|/embed/|youtu\.be/)([\w-]{11})", youtube_url)
    return m.group(1) if m else None

def fetch_transcript_text(video_id: str):
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko", "en"])
        return " ".join(x["text"] for x in srt if x["text"].strip())
    except Exception:
        return None

def split_text_by_chars(text: str, chunk_size=6000, overlap=300):
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = max(0, end - overlap)
    return chunks

def _gemini(payload: dict):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY가 설정되어 있지 않습니다."}
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + api_key
    r = requests.post(url, json=payload, timeout=120)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        return {"error": f"Gemini HTTP {r.status_code}", "body": r.text}
    return r.json()

def _extract_text(resp_json: dict) -> str:
    try:
        return resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False)

def summarize_long_text(text: str) -> str:
    chunks = split_text_by_chars(text, chunk_size=6000, overlap=300)
    partials = []
    for i, ch in enumerate(chunks, 1):
        prompt = f"[부분 {i}/{len(chunks)}] 아래 내용을 핵심 위주로 간결히 요약:\n\n{ch}"
        j = _gemini({"contents": [{"parts": [{"text": prompt}]}]})
        partials.append(f"[부분요약 {i}]\n{_extract_text(j)}")

    final_prompt = (
        "아래 부분요약들을 바탕으로 중복 없이, 논리 흐름이 있는 통합 장문 요약을 작성하라. "
        "섹션: 요지/핵심포인트/근거·데이터/반론·한계/실무적 시사점/결론 순으로 한국어로 출력.\n\n"
        + "\n\n".join(partials)
    )
    j = _gemini({"contents": [{"parts": [{"text": final_prompt}]}]})
    return _extract_text(j)

def propose_solutions(summary_or_text: str) -> str:
    prompt = (
        "너는 실전 컨설턴트다. 아래 내용을 바탕으로 실행 솔루션 7개를 제시하라. "
        "각 항목은 ①짧은 제목 ②이유(1문장) ③3단계 체크리스트 형태로, "
        "가능하면 수치 기준·마감기한·도구 예시를 포함하고 한국어로 작성.\n\n=== 내용 ===\n"
        + summary_or_text[:18000]
    )
    j = _gemini({"contents": [{"parts": [{"text": prompt}]}]})
    return _extract_text(j)

def run_pipeline(youtube_url: str):
    vid = extract_video_id(youtube_url)
    if not vid:
        return {"error": "유효한 YouTube 링크가 아님"}

    text = fetch_transcript_text(vid)
    if not text:
        return {
            "video_id": vid,
            "summary_detailed": None,
            "solutions": None,
            "error": "공개 자막이 없습니다. Whisper 단계가 필요합니다."
        }

    detailed = summarize_long_text(text)
    solutions = propose_solutions(detailed)
    return {
        "video_id": vid,
        "summary_detailed": detailed,
        "solutions": solutions
    }
