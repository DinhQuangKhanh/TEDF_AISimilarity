"""Tests for the P3 scoring refinements: paper-aligned structural layers, concept-IDF weighting,
weighted set_similarity, and configurable MDDM weights."""

from app.core import config
from app.ontology.sedo import get_sedo
from app.utils import score_calculator as sc


class Tag:
    def __init__(self, name):
        self.name = name


class Topic:
    def __init__(self, title, technologies=(), domains=()):
        self.title = title
        self.description = ""
        self.scope = ""
        self.objectives = ""
        self.expected_result = ""
        self.technologies = [Tag(t) for t in technologies]
        self.structures = []
        self.domains = [Tag(d) for d in domains]


# ── P3.1: structural excludes TaskType by default (paper §V-E) ──────────────────────
def test_structural_layers_exclude_tasktype_by_default():
    assert "TaskType" not in sc._STRUCT_LAYERS
    assert {"TechnicalStack", "Methodology"} <= sc._STRUCT_LAYERS


# ── P3.3: MDDM weights are configurable (env → tuned file → paper) ───────────────────
_PAPER = {"semantic": 0.30, "lexical": 0.20, "structure": 0.30, "domain": 0.20}
_WEIGHT_ENVS = ("MDDM_SEMANTIC", "MDDM_LEXICAL", "MDDM_STRUCTURE", "MDDM_DOMAIN", "MDDM_USE_TUNED")


def _clear_weight_env(monkeypatch):
    for k in _WEIGHT_ENVS:
        monkeypatch.delenv(k, raising=False)


def test_deployed_weights_are_a_valid_distribution():
    assert set(sc.WEIGHTS) == {"semantic", "lexical", "structure", "domain"}
    assert round(sum(sc.WEIGHTS.values()), 6) == 1.0


def test_config_falls_back_to_paper_when_unset(monkeypatch):
    _clear_weight_env(monkeypatch)
    assert config._mddm_weights() == _PAPER


def test_config_env_weights_override_and_renormalize(monkeypatch):
    _clear_weight_env(monkeypatch)
    for k in ("SEMANTIC", "LEXICAL", "STRUCTURE", "DOMAIN"):
        monkeypatch.setenv("MDDM_" + k, "1")
    w = config._mddm_weights()
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_config_uses_tuned_file_when_enabled(monkeypatch):
    import json
    import os

    _clear_weight_env(monkeypatch)
    monkeypatch.setenv("MDDM_USE_TUNED", "true")
    w = config._mddm_weights()
    path = os.path.join(config._SRC_DIR, "data", "tuned_weights.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            tuned = json.load(handle)["weights"]
        assert all(abs(w[k] - tuned[k]) < 1e-6 for k in w)   # tuned weights loaded
    else:
        assert w == _PAPER   # graceful fallback when no tuned file


def test_explicit_env_beats_tuned(monkeypatch):
    _clear_weight_env(monkeypatch)
    monkeypatch.setenv("MDDM_USE_TUNED", "true")
    for k, v in (("SEMANTIC", "0.4"), ("LEXICAL", "0.1"), ("STRUCTURE", "0.1"), ("DOMAIN", "0.4")):
        monkeypatch.setenv("MDDM_" + k, v)
    w = config._mddm_weights()
    assert abs(w["semantic"] - 0.4) < 1e-9 and abs(w["domain"] - 0.4) < 1e-9


# ── P3.2: corpus concept-IDF (rare concept weighs more than ubiquitous) ─────────────
def test_concept_idf_rare_outweighs_common():
    corpus = [Topic("App with React"), Topic("Shop with React"), Topic("Game with Unity and React")]
    idf = sc.build_concept_idf(corpus)
    assert "react" in idf and "game" in idf     # "Unity" → Game concept
    assert idf["game"] > idf["react"]           # rare concept gets a bigger weight


def test_concept_idf_ubiquitous_is_near_one():
    corpus = [Topic("React app"), Topic("React shop")]
    idf = sc.build_concept_idf(corpus)
    # react appears in all N=2 docs → log((2+1)/(2+1)) + 1 = 1.0
    assert abs(idf["react"] - 1.0) < 1e-9


# ── P3.2: weighted set_similarity ───────────────────────────────────────────────────
def test_weighted_set_similarity_downweights_common_concept():
    sedo = get_sedo()
    a, b = {"react", "kubernetes"}, {"react"}
    uniform = sedo.set_similarity(a, b, sedo.wu_palmer)
    weighted = sedo.set_similarity(a, b, sedo.wu_palmer, {"react": 1.0, "kubernetes": 5.0})
    assert weighted < uniform   # the rare, unshared kubernetes drags the weighted score down


def test_weighted_set_similarity_none_on_empty():
    sedo = get_sedo()
    assert sedo.set_similarity(set(), {"react"}, sedo.wu_palmer, {"react": 2.0}) is None


def test_weighted_set_similarity_identity_still_one():
    sedo = get_sedo()
    # sharing everything → 1.0 regardless of weights
    assert sedo.set_similarity({"react"}, {"react"}, sedo.wu_palmer, {"react": 9.0}) == 1.0


# ── P3.2: end-to-end on the structural dimension (no SBERT needed) ───────────────────
def test_ontology_dimension_weighting_reduces_base_rate_structural():
    sedo = get_sedo()
    text_a = "App with React and Kubernetes"   # shares only the ubiquitous React with B
    text_b = "App with React"
    plain = sc._ontology_dimension(sedo, text_a, text_b, sc._STRUCT_LAYERS, sedo.wu_palmer, set(), set())
    weighted = sc._ontology_dimension(sedo, text_a, text_b, sc._STRUCT_LAYERS, sedo.wu_palmer,
                                      set(), set(), {"react": 1.0, "kubernetes": 6.0})
    assert weighted < plain


def test_tasktype_excluded_from_structural_text():
    # "management system" maps to a TaskType (CRUD) concept, which must NOT feed the structural score.
    sedo = get_sedo()
    concepts = {c for c in sedo.recognize("hotel management system") if sedo.nodes[c]["layer"] in sc._STRUCT_LAYERS}
    assert "crud_management" not in concepts
