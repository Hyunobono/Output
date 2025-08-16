from __future__ import annotations
import os
from typing import List

def _chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    paras = text.split("\n")
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur += (("\n" if cur else "") + p)
        else:
            if cur: chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                cur = ""
                sent = ""
                for s in p.split(". "):
                    s2 = (s + ". ").strip()
                    if len(sent) + len(s2) <= max_chars:
                        sent += s2 + " "
                    else:
                        if sent: chunks.append(sent.strip())
                        sent = s2
                if sent: chunks.append(sent.strip())
                cur = ""
    if cur: chunks.append(cur)
    return chunks

def _summarize_with_gemini(text: str, lang: str = "ko") -> str:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
당신은 유튜브 영상 요약/분석기입니다. 입력은 영상 전체 스크립트입니다.
출력은 한국어로 아래 형식으로만 작성하세요.

[요약](불릿 5개 이내, 핵심만)
- ...
- ...

[분석](논리 구조, 주장/근거, 수사적 전략, 잠재적 편향/한계, 확인이 필요한 포인트)
- ...

텍스트:
{text}
"""
    resp = model.generate_content(prompt)
    return resp.text.strip()

def _summarize_chunks_with_gemini(chunks: List[str], lang: str = "ko") -> str:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    partials = []
    for i, ck in enumerate(chunks, 1):
        prompt = f"""
다음은 긴 원문 중 {i}/{len(chunks)} 번째 청크입니다.
이 청크의 핵심만 한국어 불릿 3~6개로 요약하세요. 불필요한 수사는 제거하세요.

청크:
{ck}
"""
        partials.append(model.generate_content(prompt).text.strip())
    merged = "\n\n".join(partials)
    final_prompt = f"""
아래는 영상 스크립트 부분요약입니다. 이를 통합해 최종 요약과 분석을 작성하세요.

요약 파트: 불릿 5개 이내
분석 파트: 논리 구조/주장-근거/수사적 전략/편향·한계/팩트체크 포인트

부분요약:
{merged}
"""
    return model.generate_content(final_prompt).text.strip()

def _summarize_with_openai(text: str, lang: str = "ko") -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
    client = OpenAI(api_key=api_key)
    prompt = f"""
다음 텍스트(유튜브 전체 스크립트)를 한국어로 요약/분석하세요.

[요약] (불릿 5개 이내)
- ...

[분석] (논리 구조, 주장/근거, 수사 전략, 편향/한계, 확인 포인트)
- ...

텍스트:
{text}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def _summarize_chunks_with_openai(chunks: List[str], lang: str = "ko") -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    partials = []
    for i, ck in enumerate(chunks, 1):
        prompt = f"""
이것은 긴 원문 중 {i}/{len(chunks)} 청크입니다.
이 청크의 핵심을 한국어 불릿 3~6개로 요약하세요.

청크:
{ck}
"""
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        partials.append(r.choices[0].message.content.strip())
    merged = "\n\n".join(partials)
    final_prompt = f"""
아래 부분요약들을 하나로 통합해 최종 요약과 분석을 한국어로 작성하세요.

[요약] 불릿 5개 이내
[분석] 논리 구조/주장-근거/수사 전략/편향·한계/확인 포인트

부분요약:
{merged}
"""
    r2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.2,
    )
    return r2.choices[0].message.content.strip()

def summarize_long_text(text: str, provider: str = None, lang: str = "ko") -> str:
    provider = (provider or os.getenv("SUMM_PROVIDER") or "gemini").lower()
    chunks = _chunk_text(text, max_chars=6000)
    if provider == "openai":
        return _summarize_with_openai(text, lang) if len(chunks) == 1 else _summarize_chunks_with_openai(chunks, lang)
    else:
        return _summarize_with_gemini(text, lang) if len(chunks) == 1 else _summarize_chunks_with_gemini(chunks, lang)

