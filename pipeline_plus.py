import os
import re
import json
import requests
import tiktoken
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 불러오기

def extract_video_id(youtube_url: str) -> str:
    u = urlparse(youtube_url)
    if u.netloc in ("youtu.be",):
        return u.path.strip("/")
    if u.netloc.endswith("youtube.com"):
        qs = parse_qs(u.query)
        if "v" in qs:
            return qs["v"][0]
    m = re.search(r"(?:v=|/embed/|youtu\\.be/)([\\w-]{11})", youtube_url)
    return m.group(1) if m else None

def fetch_transcript_text(video_id: str):
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko", "en"])
        return " ".join([x["text"] for x in srt if x["text"].strip()])
    except Exception:
        return None

def _gemini(payload):
    url = 
"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" 
+ os.environ["GEMINI_API_KEY"]
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def _extract_text(response_json):
    try:
        return response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(response_json, ensure_ascii=False)

def split_text(text, max_tokens=3000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

def summarize_long_text(text):
    chunks = split_text(text)
    partial_summaries = []

    for i, ch in enumerate(chunks, 1):
        prompt = f"[Part {i}/{len(chunks)}] 아래 내용을 간결히 요약:\n\n{ch}"
        j = _gemini({"contents": [{"parts": [{"text": prompt}]}]})
        partial_summaries.append(f"[요약 {i}]\n{_extract_text(j)}")

    final_prompt = (
        "다음은 긴 영상 내용의 부분 요약입니다. 이를 중복 없이 하나의 **구조화된 통합 
요약**으로 재정리해주세요:\n\n"
        + "\n\n".join(partial_summaries)
    )
    j = _gemini({"contents": [{"parts": [{"text": final_prompt}]}]})
    return _extract_text(j)

def run_pipeline(youtube_link):
    video_id = extract_video_id(youtube_link)
    if not video_id:
        return {"error": "Invalid YouTube link."}

    transcript_text = fetch_transcript_text(video_id)
    if not transcript_text:
        return {"error": "자막을 불러올 수 없습니다."}

    summary = summarize_long_text(transcript_text)
    return {
        "video_id": video_id,
        "summary": summary
    }

