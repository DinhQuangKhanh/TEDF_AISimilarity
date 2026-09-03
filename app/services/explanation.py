"""Module 4 decision support — the per-pair explanation report (DASSF paper §IV-A, §V-F).

Every output carries more than a score: a per-dimension breakdown, the *specific* overlapping SEDO
concepts that drove the decision, and human-readable reasons. ``build_explanation`` produces that
structured report for one pair of topics.

Note: a per-topic "revision suggestion" was intentionally removed — editing a topic is the proposing
lecturer's job, not the evaluator's. The evaluator-facing narrative is the per-field "explain
duplication" feature (``explain_service.py``).
"""

from __future__ import annotations

from app.services import capability
from app.services.preprocessing import concept_names, preprocess
from app.utils.score_calculator import (
    REASON_THRESHOLD,
    TAU_DOM,
    TAU_STR,
    action_for,
    is_structural_duplication,
    level_for,
)


def _text(thesis, *fields: str) -> str:
    return " ".join(str(getattr(thesis, f, None) or "") for f in fields).strip()


def _shared(a_ids: set[str], b_ids: set[str]) -> list[str]:
    return concept_names(a_ids & b_ids)


def build_explanation(a, b, scores: dict, capability_model=None) -> dict:
    """Build the committee-facing explanation for topic ``a`` vs matched topic ``b``.

    ``scores`` is the dict from ``calculate_scores`` (the four sub-scores + overall). ``a`` and ``b``
    are thesis-like objects exposing title/description/scope/objectives fields.
    """
    semantic = scores["semantic_score"]
    lexical = scores["lexical_score"]
    structure = scores["structure_score"]
    domain = scores["domain_score"]
    overall = scores["overall_score"]
    level = scores.get("level") or level_for(overall)

    # Concept overlap read the SAME way the dimensions do, so the "why" lines up with the scores:
    #  · structural = shared distinctive CORE FUNCTIONS (capability), read from all content fields;
    #  · domain     = shared business-domain concepts, read title-centric (SEDO);
    #  · shared tech is kept only as INFORMATION — it is NOT what drives the structural score.
    a_content = _text(a, "title", "scope", "description", "objectives", "expected_result")
    b_content = _text(b, "title", "scope", "description", "objectives", "expected_result")
    a_dom = preprocess(_text(a, "title"))
    b_dom = preprocess(_text(b, "title"))
    a_tech = preprocess(_text(a, "title", "scope", "description"))
    b_tech = preprocess(_text(b, "title", "scope", "description"))

    _stop = capability_model.get("stop") if capability_model else None
    shared_functions = sorted(capability.distinctive(a_content, _stop) & capability.distinctive(b_content, _stop))
    shared_domain = _shared(a_dom.domains, b_dom.domains)
    shared_tech = _shared(a_tech.tech, b_tech.tech)   # informational only — not the structural signal

    struct_dup = is_structural_duplication(structure, domain, overall)

    # Dimension-driven reasons, each naming the concrete evidence that fired.
    reasons: list[str] = []
    if struct_dup:
        reasons.append("same core functions with a different business domain")
    elif shared_domain and domain >= REASON_THRESHOLD:
        reasons.append(f"same business domain ({', '.join(shared_domain)})")
    if shared_functions and structure >= REASON_THRESHOLD:
        reasons.append(f"shared core functions ({', '.join(shared_functions)})")
    if lexical >= REASON_THRESHOLD:
        reasons.append("overlapping key terms")
    if semantic >= REASON_THRESHOLD:
        reasons.append("similar meaning")

    return {
        "overall_score": round(overall, 4),
        "level": level,
        "action": action_for(level),
        "is_structural_duplication": struct_dup,
        "breakdown": {
            "semantic": round(semantic, 4),
            "lexical": round(lexical, 4),
            "structure": round(structure, 4),
            "domain": round(domain, 4),
        },
        "thresholds": {"structural_tau_str": TAU_STR, "structural_tau_dom": TAU_DOM},
        "shared_concepts": {"function": shared_functions, "domain": shared_domain, "tech": shared_tech},
        "reasons": reasons,
    }
