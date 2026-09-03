"""Core-function ("capability") extraction for the structural dimension — deterministic, no LLM.

Why this exists (ICTA_REVIEW_ANALYSIS §G.2/§G.3/§H): the expert ground truth (Ground_Truth.xlsx)
judges thesis-topic duplication by the *core function set* a topic implements — booking, matching,
payment, geo-routing, ocr, … — and its rules deliberately IGNORE technology and title wording. None
of the four DASSF dimensions measured that directly: the old structural read the technology stack
(anti-correlated with the human label), and the SEDO TaskType layer is too coarse (every management
app collapses to one "CRUD" concept). Measured ceiling with the human gold tags: replacing the
structural dimension with a capability-overlap signal lifts AI↔human CV-QWK 0.456 → 0.512.

This module maps a topic's five-field text to a set of canonical capability tags via a keyword
lexicon whose vocabulary mirrors the human ``Tập chức năng cốt lõi`` labels in the ground truth
(sheet '53 De tai'). Extraction is a plain, cost-free keyword scan — no API, no model.

Matching is left-boundary (``" payment"`` matches "payment"/"payments"/"paymentgateway"): it catches
inflections while avoiding mid-word hits (``" report"`` does not fire inside "teleport").
"""

from __future__ import annotations

from functools import lru_cache

from app.utils.text_cleaner import normalize_key

# canonical capability tag -> keyword phrases (already in normalize_key form: lowercase, spaces).
# Vocabulary drawn from the recurring human labels (df ≥ 2) in Ground_Truth.xlsx, plus a few
# distinctive rare capabilities. Keep phrases specific for rare/distinctive tags (ocr, matching,
# computer vision) where a false positive is costly; generic tags (reporting, payment) carry a low
# IDF weight so occasional over-firing barely moves the score.
CAPABILITY_LEXICON: dict[str, tuple[str, ...]] = {
    "reporting": ("report", "statistic", "revenue statistic"),
    "payment": ("payment", "vnpay", "momo", "checkout", "online payment", "transaction", "e wallet", "wallet", "billing"),
    "notification": ("notification", "notify", "reminder", "push notification"),
    "review-rating": ("review", "rating", "rate and review", "feedback and rating"),
    "recommendation-ai": ("recommend", "recommendation", "personalized recommendation", "suggestion engine"),
    "order-mgmt": ("order management", "manage order", "order processing", "order tracking"),
    "booking": ("booking", "reservation", "reserve", "book online"),
    "inventory": ("inventory", "stock management", "warehouse", "stock level"),
    "marketplace-listing": ("marketplace", "product listing", "e commerce", "ecommerce", "seller", "buyer and seller"),
    "staff-mgmt": ("staff management", "employee management", "manage staff", "manage employee"),
    "chatbot-ai": ("chatbot", "chat bot", "ai assistant", "virtual assistant", "conversational"),
    "scheduling": ("scheduling", "schedule management", "auto schedule"),
    "chat": ("real time chat", "messaging", "chat box", "instant messaging", "in app chat"),
    "workflow-approval": ("approval", "approve", "workflow", "approval workflow"),
    "geo-map": ("google maps", "geolocation", "map integration", "interactive map"),
    "realtime-tracking": ("real time tracking", "live tracking", "gps tracking", "track location", "real time location"),
    "question-bank": ("question bank", "quiz", "test bank"),
    "progress-tracking": ("progress tracking", "learning progress", "track progress"),
    "geo-routing": ("routing", "route optimization", "navigation", "safe route", "shortest route", "route suggestion"),
    "grading-evaluation": ("grading", "auto grade", "automatic grading", "instant grading", "auto scoring"),
    "search-filter": ("advanced search", "search and filter", "filter by"),
    "student-mgmt": ("student management", "manage student"),
    "dashboard-analytics": ("dashboard", "analytics dashboard", "data analytics", "operations dashboard"),
    "medical-record": ("medical record", "health record", "electronic medical", "patient record"),
    "appointment-booking": ("appointment", "book appointment", "appointment booking"),
    "leaderboard": ("leaderboard", "ranking system"),
    "course-mgmt": ("course management", "manage course"),
    "computer-vision": ("computer vision", "image processing", "object detection", "image segmentation",
                        "face recognition", "background segmentation"),
    "tour-booking": ("tour booking", "travel tour", "book tour"),
    "attendance": ("attendance",),
    "alert-warning": ("alert", "warning", "early warning"),
    "risk-prediction": ("risk prediction", "risk scoring", "predict risk", "forecast", "predictive"),
    "matching": ("matchmaking", "connect drivers", "connect customers", "connect users", "match customer", "matching platform"),
    "forum-community": ("forum", "community discussion", "discussion board"),
    "maintenance-scheduling": ("maintenance",),
    "menu-mgmt": ("menu management", "manage menu", "food menu"),
    "table-booking": ("table booking", "table reservation", "book table", "reserve table"),
    "project-mgmt": ("project management", "manage project"),
    "team-mgmt": ("team management", "manage team"),
    "ocr": ("ocr", "optical character"),
    "practice-test": ("practice test", "practice exam"),
    "elearning-assessment": ("assessment", "e learning", "elearning", "online learning"),
    "content-cms": ("content management system", "cms"),
    "exam-scheduling": ("exam scheduling", "exam schedule", "schedule exam"),
    "proctoring-cv": ("proctoring", "proctor", "exam monitoring", "anti cheating"),
    "ai-feedback": ("ai feedback", "automated feedback", "ai generated feedback"),
    "mock-interview": ("mock interview", "interview practice", "practice interview"),
    "game-mechanics": ("gamification", "game mechanic", "game based"),
    "activity-tracking": ("activity tracking", "activity log"),
    "room-allocation": ("room allocation", "room assignment", "allocate room"),
    "job-matching": ("job matching", "job recommendation", "job posting", "recruitment"),
    "cv-parsing-nlp": ("cv parsing", "resume parsing", "cv analysis", "cv screening", "resume screening"),
    "semantic-search": ("semantic search", "vector search", "semantic matching"),
    "integration-api": ("api integration", "third party api"),
    "itinerary-planning": ("itinerary", "trip planning", "travel plan"),
    "document-mgmt": ("document management", "manage document"),
    # a few distinctive rare capabilities worth catching precisely
    "kyc-verification": ("kyc", "identity verification", "verify identity"),
    "delivery-tracking": ("delivery tracking", "track delivery", "shipment tracking"),
    "donation-payment": ("donation", "donate"),
    "virtual-tryon": ("virtual try on", "try on", "ar try"),
}

# Flattened (phrase, tag) index, longest phrase first so a specific phrase wins over a generic one.
_INDEX: list[tuple[str, str]] = sorted(
    ((normalize_key(kw), tag) for tag, kws in CAPABILITY_LEXICON.items() for kw in kws),
    key=lambda item: len(item[0]),
    reverse=True,
)


@lru_cache(maxsize=4096)
def extract(text: str | None) -> frozenset[str]:
    """Canonical capability tags present in ``text`` (five-field topic text). Deterministic."""
    if not text:
        return frozenset()
    padded = f" {normalize_key(text)} "
    found: set[str] = set()
    for phrase, tag in _INDEX:
        if phrase and f" {phrase}" in padded:  # left-boundary match (catches inflections)
            found.add(tag)
    return frozenset(found)


def vocabulary() -> frozenset[str]:
    return frozenset(CAPABILITY_LEXICON)


# Ubiquitous "platform functions" — plumbing almost every management app has. Used to DROP them from
# HIGHLIGHTS and from the explanation's structural evidence, so the reviewer only sees DISTINCTIVE
# shared functions (booking/matching/ocr…) rather than "both have reporting". (The scoring stoplist is
# corpus-derived per pool; this static set is a stable display filter — see highlight_service.py.)
PLATFORM_FUNCTIONS = frozenset({
    "reporting", "notification", "payment", "dashboard-analytics", "review-rating",
    "search-filter", "workflow-approval", "alert-warning", "content-cms",
})


def distinctive(text: str | None, stop=None) -> frozenset[str]:
    """Capability tags of ``text`` minus ubiquitous platform functions — the DISTINCTIVE ones.

    ``stop`` = the corpus-derived stoplist from ``build_capability_model`` (so highlights match the
    score exactly per pool); ``None`` falls back to the static ``PLATFORM_FUNCTIONS`` (single-pair use).
    """
    drop = stop if stop is not None else PLATFORM_FUNCTIONS
    return frozenset(extract(text) - drop)


def matched_surfaces(text: str | None, tags) -> list[str]:
    """Surface phrases (from the lexicon) of the given capability ``tags`` that actually appear in
    ``text`` — so a highlighter can paint the exact words that triggered a shared capability."""
    if not text or not tags:
        return []
    padded = f" {normalize_key(text)} "
    out: list[str] = []
    for tag in tags:
        for phrase in CAPABILITY_LEXICON.get(tag, ()):
            key = normalize_key(phrase)
            if key and f" {key}" in padded and phrase not in out:  # left-boundary match, as in extract()
                out.append(phrase)
    return out
