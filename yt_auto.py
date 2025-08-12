import os, sys, subprocess, tempfile
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from faster_whisper import WhisperModel
import google.generativeai as genai

# Gemini API 설정 (환경변수에서 읽음)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-1.5-flash"  # 빠르고 저렴, 요약용 최적

def get_video_id(url: str):
    q = parse_qs(urlparse(url).query)
    return q.get("v", [None])[0]

def fetch_transcript(video_id: str, langs=("ko", "en")):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        for lang in langs:
            try:
                t = transcripts.find_transcript([lang])
                data = t.fetch()
                return " ".join(x["text"].strip() for x in data if x["text"].strip())
            except Exception:
                continue
        return None
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return None

def download_audio(url: str, outdir: Path) -> Path:
    out = outdir / "audio.m4a"
    subprocess.check_call(["yt-dlp", "-f", "bestaudio", "-o", str(out), url])
    return out

def transcribe_whisper(audio_path: Path, lang="ko") -> str:
    model = WhisperModel("small", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language=lang, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)

def llm_summarize(text: str, lang="ko") -> str:
    sys_prompt = f"""You are an expert at ultra-dense summarization. Output in {lang}.
목표: 정보 손실 최소, 군더더기 제거. 고유명사/숫자/원인→결과/핵심 주장과 근거를 우선.
형식:
[초압축 요약] 2–3문장, 80–120자
[핵심 포인트] 최대 10줄, 각 줄 1문장
[검증/반론 포인트] 3–4줄
[즉시 액션] 3–4줄
스타일: 불필요 수식어·감탄사 금지, 줄마다 서로 다른 정보."""
    user_prompt = f"다음 전사에서 위 형식으로 초압축 요약을 생성하라:\n\n{text[:25000]}"
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(sys_prompt + "\n\n" + user_prompt)
    return resp.text.strip()

def main():
    if len(sys.argv) < 2:
        print("사용법: python yt_auto.py <YouTubeURL> [언어코드(기본 ko)]"); sys.exit(1)
    url = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "ko"
    vid = get_video_id(url)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        text = fetch_transcript(vid, (lang, "en")) if vid else None
        if not text:
            audio = download_audio(url, td)
            text = transcribe_whisper(audio, lang)
        out = llm_summarize(text, lang)

    outpath = Path.cwd() / f"summary_{vid or 'video'}.md"
    outpath.write_text(out, encoding="utf-8")
    print(f"✔ 완료: {outpath}")

if __name__ == "__main__":
    main()
