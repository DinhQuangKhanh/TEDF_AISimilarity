"""Per-field "explain duplication" (P5) — grounded in the SAME evidence the UI highlights.

Design guarantee: the explanation for a field is built from ``compute_highlights(query, match)`` —
the exact spans/terms/concepts the highlighter paints — so **explanation ⟷ highlight always match**.
The LLM (local Ollama, free) only rewrites that evidence into fluent Vietnamese; when no LLM is
available (or it returns junk) we fall back to a deterministic template built from the same evidence.

No "revision suggestion" here on purpose: editing the topic is the proposing lecturer's job, not the
evaluator's — this feature only *explains* the overlap.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from app.services.highlight_service import compute_highlights
from app.services.llm_client import call_llm

# Frontend field key → Vietnamese label (title = English title only; Vietnamese name is excluded).
_FIELD_LABEL = {
    "title": "Tên đề tài",
    "description": "Mô tả",
    "objectives": "Mục tiêu",
    "scope": "Phạm vi",
    "technologies": "Công nghệ",
    "expectedResults": "Kết quả mong đợi",
}


@lru_cache(maxsize=1)
def _corpus_lexical_model():
    """Same corpus-fitted TF-IDF model analyze uses, so lexical spans (hence evidence) line up exactly."""
    from app.services.corpus_loader import load_recent_capstone_corpus
    from app.services.lexical import build_lexical_scorer

    corpus = load_recent_capstone_corpus()
    return build_lexical_scorer(corpus) if corpus else None


@lru_cache(maxsize=1)
def _corpus_capability_model():
    """Same corpus capability model the score uses, so structural spans drop the SAME ubiquitous
    platform functions (per pool) instead of a static guess. None when capability is off."""
    from app.core import config
    if not config.CAPABILITY_STRUCTURAL:
        return None
    from app.services.corpus_loader import load_recent_capstone_corpus
    from app.utils.score_calculator import build_capability_model

    corpus = load_recent_capstone_corpus()
    return build_capability_model(corpus) if corpus else None


def build_field_evidence(query, match) -> list[dict]:
    """For each overlapping field, distil the highlight spans into compact evidence for one field."""
    highlights = compute_highlights(query, match, _corpus_lexical_model(), _corpus_capability_model())
    evidence: list[dict] = []
    for field in highlights.get("fields", []):
        a_spans = field.get("a", [])
        b_spans = field.get("b", [])
        a_sem = [s["text"] for s in a_spans if s["angle"] == "semantic"]
        b_sem = [s["text"] for s in b_spans if s["angle"] == "semantic"]
        terms = sorted({s["text"] for s in a_spans + b_spans if s["angle"] == "lexical"})
        concepts = sorted({s["text"] for s in a_spans + b_spans if s["angle"] in ("structural", "domain")})
        evidence.append({
            "field": field["field"],
            "label": _FIELD_LABEL.get(field["field"], field["field"]),
            "angle": field.get("angle"),
            "score": field.get("score"),
            "a": a_sem[0] if a_sem else None,   # strongest overlapping passage on the review side
            "b": b_sem[0] if b_sem else None,   # ... and on the matched side
            "terms": terms,
            "concepts": concepts,
        })
    return evidence


def _template_explanation(ev: dict) -> str:
    """Deterministic 1-line explanation from the evidence — always aligned with the highlight."""
    label = ev["label"]
    if ev["angle"] == "semantic" and ev["a"] and ev["b"]:
        pct = f"{round((ev['score'] or 0) * 100)}%"
        return (f"Hai đề tài trình bày phần {label} gần giống nhau (tương đồng ngữ nghĩa ~{pct}); "
                f"đoạn trùng rõ nhất: «{_clip(ev['a'])}» ↔ «{_clip(ev['b'])}».")
    if ev["concepts"] and ev["angle"] == "structural":
        return f"Phần {label} trùng ở các chức năng cốt lõi: {', '.join(ev['concepts'])}."
    if ev["concepts"] and ev["angle"] == "domain":
        return f"Phần {label} cùng lĩnh vực nghiệp vụ: {', '.join(ev['concepts'])}."
    if ev["concepts"]:
        return f"Phần {label} dùng chung các khái niệm: {', '.join(ev['concepts'])}."
    if ev["terms"]:
        return f"Phần {label} chia sẻ các thuật ngữ đặc trưng: {', '.join(ev['terms'])}."
    return f"Có dấu hiệu trùng ở phần {label}."


def _clip(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _prompt(evidence: list[dict]) -> str:
    slim = [{k: ev[k] for k in ("field", "label", "angle", "score", "a", "b", "terms", "concepts")} for ev in evidence]
    fields = ", ".join(ev["field"] for ev in evidence)
    return (
        "Bạn là trợ lý thẩm định đồ án tốt nghiệp. Dưới đây là BẰNG CHỨNG trùng lặp giữa hai đề tài theo từng "
        "hạng mục: 'a' và 'b' là đoạn văn giống nhau nhất ở hai bên, 'terms' là thuật ngữ chung, 'concepts' là "
        "chức năng cốt lõi/lĩnh vực chung, 'score' là mức tương đồng.\n"
        "Với MỖI hạng mục, viết 1–2 câu tiếng Việt giải thích VÌ SAO hai đề tài trùng ở hạng mục đó, "
        "CHỈ dựa vào bằng chứng cho sẵn, TUYỆT ĐỐI KHÔNG bịa thêm thông tin ngoài bằng chứng.\n"
        f"Trả về DUY NHẤT một JSON object dạng {{\"<field>\": \"<giải thích>\"}} với đúng các field: {fields}. "
        "Không kèm giải thích nào khác ngoài JSON.\n\n"
        "BẰNG CHỨNG:\n" + json.dumps(slim, ensure_ascii=False)
    )


def _parse_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def explain(query, match, use_llm: bool = True) -> list[dict]:
    """Return ``[{field, angle, score, explanation}]`` — one entry per overlapping field.

    Tries the (local, free) LLM to phrase each explanation; falls back to the deterministic template
    per field when the LLM is unavailable or its JSON is unusable. Either way the wording is grounded
    in the highlighted evidence, so it can never disagree with what the UI paints.
    """
    evidence = build_field_evidence(query, match)
    if not evidence:
        return []

    llm_map: dict | None = None
    if use_llm:
        text = call_llm(_prompt(evidence))
        if text:
            llm_map = _parse_json_object(text)

    result: list[dict] = []
    for ev in evidence:
        explanation = None
        if llm_map:
            candidate = llm_map.get(ev["field"])
            if isinstance(candidate, str) and candidate.strip():
                explanation = candidate.strip()
        if not explanation:
            explanation = _template_explanation(ev)
        result.append({
            "field": ev["field"],
            "angle": ev["angle"],
            "score": ev["score"],
            "explanation": explanation,
        })
    return result
