"""Dry-run pipeline analysis for the demo UI.

Runs the full DASSF pipeline on an ad-hoc typed topic against the seeded corpus WITHOUT
persisting anything, and returns a step-by-step trace (Module 1 → semantic backend → corpus
→ four-dimension scoring → MDDM/classification) plus the top matches with their explanation.

Each step carries a ``status`` (success / warning / error) and a human-readable ``detail`` so
the UI can paint it green / amber / red and say *why*.
"""

from __future__ import annotations

from app.services.explanation import build_explanation
from app.services.preprocessing import concept_names, preprocess
from app.utils.score_calculator import WEIGHTS, action_for, calculate_scores


def _text(obj, *fields: str) -> str:
    return " ".join(str(getattr(obj, f, None) or "") for f in fields).strip()


def _step(step_id: str, name: str, status: str, *, detail: str = "", inp=None, out=None) -> dict:
    return {"id": step_id, "name": name, "status": status, "detail": detail, "input": inp, "output": out}


def analyze_topic(query, corpus, lexical_model, *, top_k: int = 5, semantic_backend: str = "sbert") -> dict:
    """Return {steps, topMatches, weights, corpusSize} for one typed topic vs the corpus."""
    title = (getattr(query, "title", None) or "").strip()
    steps: list[dict] = []

    # ── Step 1 — Module 1: preprocessing & structured extraction ────────────────────
    if not title:
        steps.append(_step("preprocess", "Module 1 — Tiền xử lý & chuẩn hoá", "error",
                           detail="Thiếu tiêu đề đề tài — không thể phân tích."))
        return {"steps": steps, "topMatches": [], "weights": WEIGHTS, "corpusSize": len(corpus)}

    m1_full = preprocess(_text(query, "title", "scope", "description"))
    m1_title = preprocess(title)
    tech = concept_names(m1_full.tech)
    methods = concept_names(m1_full.methods)
    domains = concept_names(m1_title.domains)
    tasks = concept_names(m1_full.tasks)
    recognized = bool(tech or methods or domains or tasks)
    steps.append(_step(
        "preprocess", "Module 1 — Tiền xử lý & chuẩn hoá",
        "success" if recognized else "warning",
        detail=("Đã tách token, bỏ stop-word, chuẩn hoá và ánh xạ vào ontology SEDO."
                if recognized else
                "Tách token xong nhưng KHÔNG nhận diện được khái niệm SEDO nào (công nghệ/lĩnh vực lạ?). "
                "Chiều structural/domain có thể yếu."),
        inp={"title": title,
             "extraFields": [f for f in ("description", "scope", "objectives", "expected_result")
                             if getattr(query, f, None)]},
        out={"tokens": sorted(m1_full.tokens)[:20], "tech": tech, "methods": methods,
             "domains": domains, "tasks": tasks},
    ))

    # ── Step 2 — semantic encoder health ────────────────────────────────────────────
    sbert_ok = semantic_backend == "sbert"
    steps.append(_step(
        "semantic", "Chiều Semantic — Sentence-BERT",
        "success" if sbert_ok else "warning",
        detail=("Encoder Sentence-BERT (đa ngữ) sẵn sàng — so khớp theo *ý nghĩa*."
                if sbert_ok else
                "Chưa cài sentence-transformers → tạm lùi về token-Jaccard. "
                "Cài `sentence-transformers` để bật semantic thật."),
        out={"backend": semantic_backend},
    ))

    # ── Step 3 — corpus ──────────────────────────────────────────────────────────────
    if not corpus:
        steps.append(_step("corpus", "Nạp kho đề tài cũ", "error",
                           detail="Kho đề tài rỗng. Chạy `python seed_thesis.py` để nạp corpus trước."))
        return {"steps": steps, "topMatches": [], "weights": WEIGHTS, "corpusSize": 0}
    steps.append(_step("corpus", "Nạp kho đề tài cũ", "success",
                       detail=f"Đã nạp {len(corpus)} đề tài trước đó để đối chiếu.",
                       out={"count": len(corpus)}))

    # ── Step 4 — four-dimension scoring against every prior topic ────────────────────
    scored = [(cand, calculate_scores(query, cand, lexical_model)) for cand in corpus]
    scored.sort(key=lambda pair: pair[1]["overall_score"], reverse=True)
    steps.append(_step("scoring", "Chấm 4 chiều × từng đề tài", "success",
                       detail=f"So tiêu đề mới với {len(corpus)} đề tài trên 4 chiều "
                              f"(semantic · lexical · structural · domain).",
                       out={"compared": len(corpus),
                            "dimensions": ["semantic", "lexical", "structural", "domain"]}))

    top = scored[:top_k]
    best = top[0][1] if top else None

    # ── Step 5 — MDDM fusion + classification ────────────────────────────────────────
    steps.append(_step("mddm", "Gộp điểm MDDM + phân loại mức", "success",
                       detail=(f"Đề tài giống nhất được gộp thành {best['overall_score']:.2f} → mức "
                               f"'{best['level']}' ({action_for(best['level'])})."
                               if best else "Không có đề tài để so."),
                       out={"weights": WEIGHTS,
                            "topOverall": round(best["overall_score"], 4) if best else None,
                            "topLevel": best["level"] if best else None,
                            "topAction": action_for(best["level"]) if best else None}))

    matches = []
    for cand, scores in top:
        report = build_explanation(query, cand, scores)
        report["otherTitle"] = cand.title
        report["otherSemester"] = getattr(cand, "semester", None)
        matches.append(report)

    return {"steps": steps, "topMatches": matches, "weights": WEIGHTS, "corpusSize": len(corpus)}
