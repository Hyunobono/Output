import argparse, os
from transcript_pipeline import (
    fetch_transcript_if_available,
    transcript_to_plaintext,
    download_audio,
    whisper_transcribe_local,
    whisper_to_plaintext,
)
from summarizer import summarize_long_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube URL")
    ap.add_argument("--lang", default="ko", help="summary language (ko/en...)")
    ap.add_argument("--provider", default=os.getenv("SUMM_PROVIDER", "gemini"), help="gemini|openai")
    ap.add_argument("--force_whisper", action="store_true", help="항상 Whisper 사용(자막 무시)")
    args = ap.parse_args()

    print("▶ 텍스트 추출 중...")
    text = None
    if not args.force_whisper:
        trans = fetch_transcript_if_available(args.url, [args.lang, "en"])
        if trans:
            text = transcript_to_plaintext(trans)

    if not text:
        print("⚠ 자막 실패 → Whisper 전환")
        audio = download_audio(args.url, "out")
        segs = whisper_transcribe_local(audio, language_hint=args.lang)
        text = whisper_to_plaintext(segs)

    print(f"✓ 텍스트 길이: {len(text):,} chars")
    print("▶ 요약/분석 중...")
    out = summarize_long_text(text, provider=args.provider, lang=args.lang)
    print("\n===== 결과 =====\n")
    print(out)

if __name__ == "__main__":
    main()

