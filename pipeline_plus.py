import os
import json
import re
import requests
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()  # .env에서 환경변수 불러오기

def extract_video_id(youtube_url: str) -> str:
    """YouTube 링크에서 영상 ID 추출"""
    u = urlparse(youtube_url)
    if u.hostname in ("youtu.be",):
        return u.path.lstrip("/")
    if u.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        v = parse_qs(u.query).get("v", [None])[0]
        if v:
            return v
    m = re.search(r"(?:v=|/embed/|youtu\.be/)([\w-]{11})", youtube_url)
    return m.group(1) if m else None

def fetch_transcript_text(video_id: str):
    """공개 자막 가져오기(ko 우선, 없으면 en)"""
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko", "en"])
        return " ".join(x["text"] for x in srt if x["text"].strip())
    except Exception:
        return None

def split_text_by_chars(text: str, chunk_size=6000, overlap=300):
    """긴 텍스트를 문자 길이 기준으로 나눔(의존성 최소화)"""
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
    """Google Gemini API 호출(에러 메시지 안전 반환)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY가 설정되어 있지 않습니다."}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        return {"error": f"Gemini HTTP {r.status_code}", "body": r.text}
    except Exception as e:
        return {"error": f"Gemini 요청 실패: {e}"}

def _extract_text(resp_json: dict) -> str:
    """Gemini 응답에서 텍스트만 추출(실패 시 원문 반환)"""
    try:
        return resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False)

def summarize_long_text(text: str) -> str:
    """긴 텍스트를 분할 요약 후 통합 요약(모든 프롬프트 3중 따옴표 사용)"""
    chunks = split_text_by_chars(text, chunk_size=6000, overlap=300)
    partials = []
    for i, ch in enumerate(chunks, 1):
        prompt = f"""[부분 {i}/{len(chunks)}]
아래 내용을 핵심 위주로 간결히 요약하라.

{ch}
"""
        j = _gemini({"contents": [{"parts": [{"text": prompt}]}]})
        partials.append(f"[부분요약 {i}]\n{_extract_text(j)}")

    final_prompt = """아래 '부분요약'들을 바탕으로 중복 없이, 논리 흐름이 있는 통합 장문 요약을 작성하라.
섹션: 요지 / 핵심포인트 / 근거·데이터 / 반론·한계 / 실무적 시사점 / 결론
한국어로 출력하라.

""" + "\n\n".join(partials)

    j = _gemini({"contents": [{"parts": [{"text": final_prompt}]}]})
    return _extract_text(j)

def propose_solutions(summary_or_text: str) -> str:
    """요약문 기반 실행 솔루션 7개 제시(3중 따옴표)"""
    prompt = f"""너는 실전 컨설턴트다. 아래 내용을 바탕으로 실행 솔루션 7개를 제시하라.
각 항목은 ①짧은 제목 ②이유(1문장) ③3단계 체크리스트 형태로,
가능하면 수치 기준·마감기한·도구 예시를 포함하고 한국어로 작성.

=== 내용 ===
{summary_or_text[:18000]}
"""
    j = _gemini({"contents": [{"parts": [{"text": prompt}]}]})
    return _extract_text(j)

def run_pipeline(youtube_url: str):
    """전체 파이프라인 실행"""
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
