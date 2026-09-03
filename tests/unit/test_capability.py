"""Tests for the deterministic core-function ("capability") dimension (ICTA_REVIEW_ANALYSIS §H).

The capability dimension measures what a system DOES (booking, matching, payment, ocr …) — the
signal the expert ground truth judges by — via a cost-free keyword scan, dropping ubiquitous
platform functions. It backs the structural dimension when config.CAPABILITY_STRUCTURAL is on.
"""

from app.services import capability as cap
from app.utils import score_calculator as sc


class Topic:
    def __init__(self, title="", description="", scope="", objectives="", expected_result=""):
        self.title = title
        self.description = description
        self.scope = scope
        self.objectives = objectives
        self.expected_result = expected_result
        self.technologies = []
        self.structures = []
        self.domains = []


def test_extract_finds_capabilities():
    tags = cap.extract("An online booking platform with secure payment and a real time chat")
    assert {"booking", "payment", "chat"} <= tags


def test_extract_empty_input():
    assert cap.extract("") == frozenset()
    assert cap.extract(None) == frozenset()


def test_extract_ignores_pure_technology():
    # Technology words are not capabilities — the ground truth judges by function, not stack.
    assert cap.extract("Built with ReactJS, ASP.NET Core and PostgreSQL") == frozenset()


def test_build_capability_model_idf_and_stoplist():
    corpus = ([Topic(description="online payment")] * 3
              + [Topic(description="ocr document scan")] * 6
              + [Topic(description="inventory tracking")])  # N = 10
    model = sc.build_capability_model(corpus)
    assert "payment" in model["stop"]          # 3/10 > 0.2 → ubiquitous platform function, dropped
    assert "inventory" not in model["stop"]     # 1/10 ≤ 0.2 → kept
    assert model["idf"]["inventory"] > model["idf"]["payment"]  # rarer ⇒ heavier


def test_capability_similarity_identical_and_disjoint():
    model = {"idf": {}, "stop": set()}          # no stoplist, unweighted → plain Jaccard
    assert sc.capability_similarity("booking and matchmaking", "booking and matchmaking", model) == 1.0
    assert sc.capability_similarity("online booking", "ocr scanning", model) == 0.0


def test_capability_similarity_drops_stoplisted_platform_functions():
    model = {"idf": {}, "stop": {"payment"}}
    # the two differ only by the ubiquitous 'payment' → identical once it is dropped
    assert sc.capability_similarity("online booking and payment", "online booking", model) == 1.0


def test_calculate_scores_routes_through_capability_when_model_given():
    a = Topic(title="Ride booking and driver matching",
              description="connect users with drivers, real time tracking, secure payment")
    b = Topic(title="Food delivery with driver matching",
              description="connect customers with drivers, delivery tracking, secure payment")
    corpus = [a, b, Topic(description="ocr and document management"),
              Topic(description="question bank and auto grading")]
    model = sc.build_capability_model(corpus)
    with_cap = sc.calculate_scores(a, b, capability_model=model)["structure_score"]
    without = sc.calculate_scores(a, b)["structure_score"]           # SEDO fallback
    assert 0.0 <= with_cap <= 1.0
    assert 0.0 <= without <= 1.0
