"""Dry-run pipeline analysis for the demo UI.

Runs the full DASSF pipeline on an ad-hoc typed topic against the seeded corpus WITHOUT
persisting anything, and returns a **step-by-step trace** the UI (and ``tools/trace_pipeline.py``)
can replay:

    1 input      nhận đề tài                → các trường đã điền, chế độ so khớp
    2 preprocess Module 1 — tiền xử lý       → tập token đã chuẩn hoá
    3 sedo       Module 1 — NER ontology     → (tech, method, domain, task)
    4 semantic   nạp encoder ngữ nghĩa       → tên backend
    5 corpus     nạp kho + khớp mô hình      → N đề tài, backend TF-IDF, concept-IDF
    6 scoring    chấm 4 chiều × N đề tài     → ma trận điểm, top-K
    7 mddm       gộp điểm (Eq. 1)            → từng số hạng α·S_sem … và tổng
    8 decision   ngưỡng Table III + luật     → mức, hành động, cờ trùng cấu trúc

Each step carries a ``status`` (success / warning / error), a human-readable ``detail`` so the UI
can paint it green / amber / red and say *why*, and an explicit ``input`` / ``output`` pair (plus a
``formula`` where the step is arithmetic) so the reader can follow the data, not just the verdict.
"""

from __future__ import annotations

from app.core import config
from app.core.config import MDDM_WEIGHTS_SOURCE, WPATH_K
from app.services.explanation import build_explanation
from app.services.highlight_service import compute_highlights
from app.services.preprocessing import concept_names, preprocess
from app.utils.score_calculator import (
    LEVEL_THRESHOLDS,
    TAU_DOM,
    TAU_STR,
    WEIGHTS,
    action_for,
    build_capability_model,
    build_concept_idf,
    calculate_scores,
)

_CONTENT_FIELDS = ("title", "description", "scope", "objectives", "expected_result")
_GREEK = {"semantic": "α", "lexical": "β", "structure": "γ", "domain": "δ"}
_DIM_FORMULAS = {
    "semantic": "S_sem = cosine(MiniLM(A'), MiniLM(B')) — A',B' = văn bản đã CHE tên công nghệ; so theo *ý nghĩa nghiệp vụ*",
    "lexical": "S_lex = cosine(TF-IDF(A), TF-IDF(B)) — so theo *từ ngữ đặc trưng*",
    "structure": "S_str = Wu-Palmer trên SEDO {TaskType} (loại chức năng hệ thống LÀM) — so theo *chức năng*, KHÔNG lấy công nghệ",
    "domain": f"S_dom = wpath(k={WPATH_K}) trên SEDO {{DomainEntity}} — so theo *lĩnh vực nghiệp vụ*",
}


def _structure_formula() -> str:
    """The structural dimension's live formula — capability overlap when CAPABILITY_STRUCTURAL is on
    (the deployed default), else the coarse SEDO TaskType."""
    if config.CAPABILITY_STRUCTURAL:
        return ("S_str = Jaccard-IDF trên TẬP CHỨC NĂNG CỐT LÕI (booking/matching/attendance/ocr…), "
                "bỏ chức năng nền tảng phổ biến — so theo *chức năng thật*, KHÔNG lấy công nghệ")
    return _DIM_FORMULAS["structure"]


def _content_fields(topic) -> dict:
    """The six comparable fields of a topic, for the side-by-side view."""
    return {
        "title": getattr(topic, "title", None),
        "description": getattr(topic, "description", None),
        "scope": getattr(topic, "scope", None),
        "objectives": getattr(topic, "objectives", None),
        "expected_result": getattr(topic, "expected_result", None),
        "technologies": [t.name for t in getattr(topic, "technologies", [])],
    }


def _text(obj, *fields: str) -> str:
    return " ".join(str(getattr(obj, f, None) or "") for f in fields).strip()


def _step(step_id: str, name: str, status: str, *, detail: str = "", inp=None, out=None,
          formula: str | list[str] | None = None) -> dict:
    return {"id": step_id, "name": name, "status": status, "detail": detail,
            "input": inp, "output": out, "formula": formula}


def _has_body(topic) -> bool:
    return any((getattr(topic, f, None) or "").strip()
               for f in ("description", "scope", "objectives", "expected_result"))


def analyze_topic(query, corpus, lexical_model, *, top_k: int = 5, semantic_backend: str = "sbert") -> dict:
    """Return {steps, topMatches, weights, weightsSource, corpusSize, query} for one typed topic."""
    title = (getattr(query, "title", None) or "").strip()
    steps: list[dict] = []
    stop = {"steps": steps, "topMatches": [], "weights": WEIGHTS,
            "weightsSource": MDDM_WEIGHTS_SOURCE, "corpusSize": len(corpus)}

    # ── Bước 1 — nhận đề tài ─────────────────────────────────────────────────────────
    filled = {f: (getattr(query, f, None) or "").strip() for f in _CONTENT_FIELDS}
    techs = [t.name for t in getattr(query, "technologies", [])]
    if not title:
        steps.append(_step("input", "Nhận đề tài cần kiểm tra", "error",
                           detail="Thiếu tiêu đề đề tài — không thể phân tích.",
                           inp={"fields": {k: len(v) for k, v in filled.items()}}))
        return stop
    full_content = _has_body(query)
    steps.append(_step(
        "input", "Nhận đề tài cần kiểm tra", "success" if full_content else "warning",
        detail=("Đủ nội dung → so khớp toàn bộ 5 trường + công nghệ." if full_content else
                "Chỉ có tiêu đề → hệ thống lùi về so tiêu đề-với-tiêu đề cho công bằng. "
                "Điền thêm mô tả/phạm vi/mục tiêu để kết quả chính xác hơn."),
        inp={"fields": {k: len(v) for k, v in filled.items()}, "technologies": techs},
        out={"filledFields": [k for k, v in filled.items() if v],
             "technologies": techs,
             "totalChars": sum(len(v) for v in filled.values()),
             "matchMode": "full-content" if full_content else "title-only"},
    ))

    # ── Bước 2 — Module 1: tiền xử lý & chuẩn hoá ────────────────────────────────────
    m1_source = _text(query, "title", "scope", "description")
    m1_full = preprocess(m1_source)
    m1_title = preprocess(title)
    steps.append(_step(
        "preprocess", "Module 1 — Tiền xử lý & chuẩn hoá", "success",
        detail=f"Tách {len(m1_full.tokens)} token có nghĩa từ {len(m1_source)} ký tự đầu vào.",
        inp={"text": m1_source[:280], "chars": len(m1_source)},
        out={"tokens": sorted(m1_full.tokens), "tokenCount": len(m1_full.tokens)},
        formula=["1) tách token", "2) bỏ stop-word (Anh + Việt)", "3) đưa về số ít (lemma)",
                 "4) gộp đồng nghĩa bề mặt (reactjs / react.js → react)"],
    ))

    # ── Bước 3 — Module 1: nhận diện khái niệm SEDO ──────────────────────────────────
    tech = concept_names(m1_full.tech)
    methods = concept_names(m1_full.methods)
    domains = concept_names(m1_title.domains)   # lĩnh vực đọc từ TIÊU ĐỀ (xem score_calculator)
    tasks = concept_names(m1_full.tasks)
    recognized = bool(tech or methods or domains or tasks)
    steps.append(_step(
        "sedo", "Module 1 — Nhận diện khái niệm SEDO", "success" if recognized else "warning",
        detail=(f"Ánh xạ được {len(tech) + len(methods) + len(domains) + len(tasks)} khái niệm "
                f"vào ontology SEDO." if recognized else
                "KHÔNG nhận diện được khái niệm SEDO nào (công nghệ/lĩnh vực lạ?). "
                "Chiều structural/domain sẽ yếu — điểm hai chiều này có thể về 0."),
        inp={"tokens": sorted(m1_full.tokens)[:20],
             "domainSource": "chỉ tiêu đề (mô tả bị loại để không nhiễu lĩnh vực)"},
        out={"tech": tech, "methods": methods, "domains": domains, "tasks": tasks},
        formula="văn bản → (TechnicalStack, Methodology, DomainEntity, TaskType)",
    ))

    # ── Bước 4 — encoder ngữ nghĩa ───────────────────────────────────────────────────
    sbert_ok = semantic_backend not in ("jaccard-fallback", "")
    steps.append(_step(
        "semantic", "Nạp encoder ngữ nghĩa", "success" if sbert_ok else "warning",
        detail=(f"Encoder `{semantic_backend}` sẵn sàng — so khớp theo *ý nghĩa*, "
                f"không cần trùng từ." if sbert_ok else
                "Chưa cài fastembed → tạm lùi về token-Jaccard. Đây KHÔNG phải semantic thật; "
                "cài `fastembed` để bật encoder."),
        inp={"text": (title[:120] + "…") if len(title) > 120 else title},
        out={"backend": semantic_backend},
        formula="che tên công nghệ → text'; vector = encode(text'); S_sem = cosine(v_A, v_B)",
    ))

    # ── Bước 5 — kho đề tài + mô hình mức corpus ─────────────────────────────────────
    if not corpus:
        steps.append(_step("corpus", "Nạp kho đề tài & khớp mô hình corpus", "error",
                           detail="Kho đề tài rỗng. Chạy `python seed_thesis.py` để nạp corpus trước.",
                           out={"count": 0}))
        stop["corpusSize"] = 0
        return stop

    concept_idf = build_concept_idf(corpus)
    capability_model = build_capability_model(corpus) if config.CAPABILITY_STRUCTURAL else None
    lexical_backend = getattr(lexical_model, "backend", None) or "weighted-jaccard-fallback"
    tfidf_ok = lexical_backend == "tfidf-cosine"
    semesters = sorted({s for s in (getattr(t, "semester", None) for t in corpus) if s})
    steps.append(_step(
        "corpus", "Nạp kho đề tài & khớp mô hình corpus", "success" if tfidf_ok else "warning",
        detail=(f"Đã nạp {len(corpus)} đề tài và khớp TF-IDF + IDF khái niệm trên chính kho này."
                if tfidf_ok else
                f"Đã nạp {len(corpus)} đề tài, nhưng scikit-learn không có → chiều lexical lùi về "
                f"Jaccard có trọng số (không phải TF-IDF cosine như bài báo). "
                f"Cài `scikit-learn` để bật đúng công thức."),
        inp={"source": "kho đề tài các kỳ gần nhất"},
        out={"count": len(corpus), "semesters": semesters,
             "lexicalBackend": lexical_backend, "conceptIdfSize": len(concept_idf)},
        formula="IDF(khái niệm) = log((N+1)/(df+1)) + 1 → khái niệm phổ biến (React, CRUD) bị hạ trọng số",
    ))

    # ── Bước 6 — chấm 4 chiều với từng đề tài trong kho ──────────────────────────────
    scored = [(cand, calculate_scores(query, cand, lexical_model, concept_idf, capability_model)) for cand in corpus]
    scored.sort(key=lambda pair: pair[1]["overall_score"], reverse=True)
    top = scored[:top_k]
    best = top[0][1] if top else None
    preview = [{"title": cand.title,
                "semantic": s["semantic_score"], "lexical": s["lexical_score"],
                "structure": s["structure_score"], "domain": s["domain_score"],
                "overall": s["overall_score"]} for cand, s in top]
    steps.append(_step(
        "scoring", "Chấm 4 chiều × từng đề tài trong kho", "success",
        detail=f"So đề tài của bạn với {len(corpus)} đề tài cũ, mỗi cặp cho ra 4 điểm độc lập "
               f"→ {len(corpus) * 4} phép chấm, rồi xếp hạng theo điểm gộp.",
        inp={"pairs": len(corpus), "dimensions": list(_DIM_FORMULAS)},
        out={"compared": len(corpus), "topK": len(top), "rows": preview},
        formula=[_DIM_FORMULAS["semantic"], _DIM_FORMULAS["lexical"], _structure_formula(), _DIM_FORMULAS["domain"]],
    ))

    # ── Bước 7 — gộp điểm MDDM (Eq. 1) ───────────────────────────────────────────────
    terms = []
    if best:
        for dim, key in (("semantic", "semantic_score"), ("lexical", "lexical_score"),
                         ("structure", "structure_score"), ("domain", "domain_score")):
            terms.append({"dim": dim, "symbol": _GREEK[dim], "weight": round(WEIGHTS[dim], 4),
                          "score": round(best[key], 4),
                          "product": round(WEIGHTS[dim] * best[key], 4)})
        expression = " + ".join(f"{t['weight']:.2f}×{t['score']:.2f}" for t in terms)
        expression += f" = {best['overall_score']:.4f}"
    else:
        expression = None
    steps.append(_step(
        "mddm", "Gộp điểm MDDM (Eq. 1)", "success",
        detail=(f"Bốn điểm của đề tài giống nhất được gộp bằng trọng số đã học "
                f"→ {best['overall_score']:.4f}." if best else "Không có đề tài để so."),
        inp={"scores": {t["dim"]: t["score"] for t in terms},
             "weights": WEIGHTS, "weightsSource": MDDM_WEIGHTS_SOURCE},
        out={"terms": terms, "expression": expression,
             "overall": round(best["overall_score"], 4) if best else None},
        formula="S_composite = α·S_sem + β·S_lex + γ·S_str + δ·S_dom   (α+β+γ+δ = 1)",
    ))

    # ── Bước 8 — phân mức & quyết định ───────────────────────────────────────────────
    thresholds = [{"min": lo, "level": lv} for lo, lv in LEVEL_THRESHOLDS]
    if best:
        struct_dup = best["structural_duplication"]
        decision_detail = (f"{best['overall_score']:.4f} → mức '{best['level']}' "
                           f"({action_for(best['level'])}).")
        if struct_dup:
            decision_detail += (f" Ngoài ra S_str={best['structure_score']:.2f} ≥ {TAU_STR}, "
                                f"S_dom={best['domain_score']:.2f} < {TAU_DOM} và mức ≥ High "
                                f"→ CÙNG CHỨC NĂNG LÕI, KHÁC LĨNH VỰC.")
    else:
        struct_dup, decision_detail = False, "Không có đề tài để so."
    steps.append(_step(
        "decision", "Phân mức & khuyến nghị xử lý", "success",
        detail=decision_detail,
        inp={"overall": round(best["overall_score"], 4) if best else None,
             "structure": round(best["structure_score"], 4) if best else None,
             "domain": round(best["domain_score"], 4) if best else None},
        out={"level": best["level"] if best else None,
             "action": action_for(best["level"]) if best else None,
             "structuralDuplication": struct_dup,
             "thresholds": thresholds,
             "structuralRule": {"tauStr": TAU_STR, "tauDom": TAU_DOM, "minLevel": "High"}},
        formula=["mức (ngưỡng calibrate): " + "  ".join(f"≥{lo:.2f}→{lv}" for lo, lv in LEVEL_THRESHOLDS),
                 f"trùng cấu trúc: S_str ≥ {TAU_STR} VÀ S_dom < {TAU_DOM} VÀ mức ≥ High"],
    ))

    matches = []
    for cand, scores in top:
        report = build_explanation(query, cand, scores, capability_model)
        report["otherTitle"] = cand.title
        report["otherSemester"] = getattr(cand, "semester", None)
        report["other"] = _content_fields(cand)          # matched topic content, for side-by-side
        report["highlights"] = compute_highlights(query, cand, lexical_model, capability_model)   # field-aligned overlap spans
        matches.append(report)

    return {
        "steps": steps,
        "topMatches": matches,
        "weights": WEIGHTS,
        "weightsSource": MDDM_WEIGHTS_SOURCE,
        "corpusSize": len(corpus),
        "query": _content_fields(query),                  # echo so the UI has both sides
    }
