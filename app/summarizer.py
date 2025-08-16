from __future__ import annotations
import google.generativeai as genai

DEFAULT_MODEL = "gemini-1.5-flash"

def config_gemini(api_key: str, model: str = DEFAULT_MODEL):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)

def chunk_text(text: str, max_chars: int = 8000):
    blocks, cur, n = [], 0, len(text)
    while cur < n:
        blocks.append(text[cur: cur + max_chars])
        cur += max_chars
    return blocks

PART_PROMPT = """You are an expert note-taker.
Summarize the following transcript chunk in Korean with bullet points, keeping names/figures.
Keep it concise but information-dense.
Text:
```{chunk}```"""

FINAL_PROMPT = """You are an expert summarizer.
Merge these partial summaries into one cohesive Korean summary with:
- 7~12 bullet points (핵심요지)
- 5줄 이내 한글 요약문 (요지정리)
- 10~15개의 키워드 태그 (#태그 형식, 영문)
Partial summaries:
```{partials}```"""

def summarize_long_text(api_key: str, full_text: str) -> dict:
    model = config_gemini(api_key)
    blocks = chunk_text(full_text, max_chars=8000)

    partials = []
    for b in blocks:
        resp = model.generate_content(PART_PROMPT.format(chunk=b))
        partials.append(resp.text.strip())

    final = model.generate_content(
        FINAL_PROMPT.format(partials="\n\n".join(partials))
    ).text.strip()

    return {"partials": partials, "final": final}
