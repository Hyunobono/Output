import os, json, re, requests, subprocess, tempfile
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()

# ── 유튜브 ID 추출 ─────────────────────────────────
def extract_video_id(youtube_url: str) -> str:
    u = urlparse(youtube_url)
    host = (u.hostname or "").lower()
    if host == "youtu.be":
        return u.path.lstrip("/")
    if host in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        v = parse_qs(u.query).get("v", [None])[0]
        if v: return v
    m = re.search(r"(?:v=|/embed/|youtu\.be/)([\w-]{11})", youtube_url)
    return m.group(1) if m else None

# ── 공개 자막 가져오기 ─────────────────────────────
def fetch_transcript_text(video_id: str):
    try:
        srt = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko","en"])
        return " ".join(x["text"] for x in srt if x["text"].strip())
    except Exception:
        return None

# ── (자막 없을 때) 오디오 다운로드 + Whisper 전사 ───────
def _download_audio_with_ytdlp(youtube_url: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    out = os.path.join(tmpdir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f","bestaudio/best",
        "--extract-audio",
        "--audio-format","m4a",
        "--audio-quality","0",
        "-o", out,
        youtube_url,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    for fn in os.listdir(tmpdir):
        if fn.endswith(".m4a"):
            return os.path.join(tmpdir, fn)
    raise RuntimeError("오디오 파일을 찾지 못했습니다.")

def _whisper_transcribe_local(audio_path: str, model_size="base") -> str:
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)  # 언어 자동 감지
    return result.get("text","").strip()

# ── 긴 텍스트 나누기 ───────────────────────────────
def split_text_by_chars(text: str, chunk_size=6000, overlap=300):
    chunks=[]; i=0; n=len(text)
    while i<n:
        end=min(i+chunk_size,n)
        chunks.append(text[i:end])
        if end==n: break
        i=max(0,end-overlap)
    return chunks

# ── Gemini 호출 ───────────────────────────────────
def _gemini(payload: dict):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error":"GEMINI_API_KEY not set"}
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key="+api_key
    try:
        r=requests.post(url,json=payload,timeout=120); r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        return {"error":f"Gemini HTTP {r.status_code}","body":r.text}
    except Exception as e:
        return {"error":f"Gemini request failed: {e}"}

def _extract_text(resp_json: dict) -> str:
    try:
        return resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(resp_json,ensure_ascii=False)

# ── 요약(부분 → 통합) ─────────────────────────────
def summarize_long_text(text: str) -> str:
    chunks = split_text_by_chars(text, chunk_size=6000, overlap=300)
    partials=[]
    for i,ch in enumerate(chunks,1):
        prompt="\n".join([
            f"[부분 {i}/{len(chunks)}]",
            "아래 내용을 핵심 위주로 간결히 요약하라.",
            "",
            ch
        ])
        j=_gemini({"contents":[{"parts":[{"text":prompt}]}]})
        partials.append("[부분요약 {}]\n{}".format(i,_extract_text(j)))
    final_prompt="\n".join([
        "아래 '부분요약'들을 바탕으로 중복 없이, 논리 흐름이 있는 통합 장문 요약을 작성하라.",
        "섹션: 요지 / 핵심포인트 / 근거·데이터 / 반론·한계 / 실무적 시사점 / 결론",
        "한국어로 출력하라.",
        "",
        "\n\n".join(partials)
    ])
    j=_gemini({"contents":[{"parts":[{"text":final_prompt}]}]})
    return _extract_text(j)

# ── 실행 솔루션 7개 ──────────────────────────────
def propose_solutions(summary_or_text: str) -> str:
    prompt="\n".join([
        "너는 실전 컨설턴트다. 아래 내용을 바탕으로 실행 솔루션 7개를 제시하라.",
        "각 항목은 ①짧은 제목 ②이유(1문장) ③3단계 체크리스트 형태로,",
        "가능하면 수치 기준·마감기한·도구 예시를 포함하고 한국어로 작성.",
        "",
        "=== 내용 ===",
        summary_or_text[:18000]
    ])
    j=_gemini({"contents":[{"parts":[{"text":prompt}]}]})
    return _extract_text(j)

# ── 엔트리 ───────────────────────────────────────
def run_pipeline(youtube_url: str):
    vid = extract_video_id(youtube_url)
    if not vid:
        return {"error":"유효한 YouTube 링크가 아님"}

    text = fetch_transcript_text(vid)
    if not text:
        # 공개 자막 없으면 Whisper 자동 실행 (로컬)
        try:
            audio = _download_audio_with_ytdlp(youtube_url)
            text  = _whisper_transcribe_local(audio, model_size="base")
        except Exception as e:
            return {
                "video_id": vid,
                "summary_detailed": None,
                "solutions": None,
                "error": f"공개 자막 없음 + Whisper 실패: {e}"
            }

    detailed  = summarize_long_text(text)
    solutions = propose_solutions(detailed)
    return {"video_id":vid,"summary_detailed":detailed,"solutions":solutions}
