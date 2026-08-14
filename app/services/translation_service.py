"""On-demand LLM translation of a topic's content, for the side-by-side duplicate-check view.

The similarity corpus is mostly English but proposed topics are Vietnamese; translating a matched
English topic into Vietnamese lets a reviewer compare in one language, and lets the UI highlight
overlapping domain terms that would otherwise never match across languages.

Falls back to the original text when the LLM is unavailable (no OPENAI_API_KEY) or errors out.
"""

import json

from app.services.llm_normalizer_service import OPENAI_API_KEY, _call_openai

_FIELDS = ("title", "description", "scope", "objectives", "expected_result")


def translate_to_vietnamese(content: dict) -> dict:
    """Translate the five content fields to Vietnamese.

    Returns the same keys plus ``translated`` (False when the LLM is unavailable or fails).
    """
    fields = {key: content.get(key) for key in _FIELDS}
    if not OPENAI_API_KEY or not any(fields.values()):
        return {**fields, "translated": False}

    prompt = (
        "Translate the following software-thesis fields from English to Vietnamese. "
        "Keep technology names and proper nouns (React, Node.js, MongoDB, FPT, ...) unchanged. "
        "Do not add explanations. Return ONLY a JSON object with exactly these keys "
        f"({', '.join(_FIELDS)}) and their translated values.\n\n"
        + json.dumps(fields, ensure_ascii=False)
    )

    try:
        raw = _call_openai(prompt)
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else {}
        return {**{key: (parsed.get(key) or fields[key]) for key in _FIELDS}, "translated": bool(parsed)}
    except Exception:
        return {**fields, "translated": False}
