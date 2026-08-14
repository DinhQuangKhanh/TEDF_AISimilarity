"""SEDO ontology: loader + Wu-Palmer / wpath similarity measures.

Backs the structural and domain dimensions of DASSF (paper Sect. 3.3-3.4).
The ontology data lives in ``sedo.json`` (a DRAFT that the author must replace
with the real SEDO). Only the ``isA`` (parent) relation is used by the measures.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from app.utils.text_cleaner import normalize_key

_SEDO_PATH = Path(__file__).with_name("sedo.json")


class Sedo:
    def __init__(self, concepts: list[dict]):
        self.nodes: dict[str, dict] = {c["id"]: c for c in concepts}
        self.children: dict[str, list[str]] = {cid: [] for cid in self.nodes}
        for node in concepts:
            parent = node.get("parent")
            if parent and parent in self.children:
                self.children[parent].append(node["id"])
        self.total = len(self.nodes)
        # keyword -> concept id, longest keywords first so specific terms win
        self._keyword_index: list[tuple[str, str]] = []
        for node in concepts:
            for keyword in node.get("keywords", []):
                key = normalize_key(keyword)
                if key:
                    self._keyword_index.append((key, node["id"]))
        self._keyword_index.sort(key=lambda item: len(item[0]), reverse=True)

    # ---- tree helpers -------------------------------------------------- #
    @lru_cache(maxsize=None)
    def depth(self, concept_id: str) -> int:
        """Distance from the root (root = 1)."""
        node = self.nodes.get(concept_id)
        if node is None:
            return 0
        parent = node.get("parent")
        if not parent:
            return 1
        return self.depth(parent) + 1

    @lru_cache(maxsize=None)
    def ancestors(self, concept_id: str) -> tuple[str, ...]:
        """Concept id then its ancestors up to the root."""
        chain: list[str] = []
        current: str | None = concept_id
        while current:
            chain.append(current)
            current = self.nodes.get(current, {}).get("parent")
        return tuple(chain)

    @lru_cache(maxsize=None)
    def descendant_count(self, concept_id: str) -> int:
        total = 0
        for child in self.children.get(concept_id, []):
            total += 1 + self.descendant_count(child)
        return total

    def lcs(self, c1: str, c2: str) -> str | None:
        """Least common subsumer (deepest shared ancestor)."""
        a2 = set(self.ancestors(c2))
        for ancestor in self.ancestors(c1):  # nearest first
            if ancestor in a2:
                return ancestor
        return None

    def path_length(self, c1: str, c2: str) -> int:
        subsumer = self.lcs(c1, c2)
        if subsumer is None:
            return 0
        return self.depth(c1) + self.depth(c2) - 2 * self.depth(subsumer)

    def information_content(self, concept_id: str) -> float:
        """Intrinsic IC (Seco et al.): 1 - log(hypo+1)/log(N)."""
        hypo = self.descendant_count(concept_id)
        return 1.0 - math.log(hypo + 1) / math.log(self.total)

    # ---- similarity measures ------------------------------------------ #
    def wu_palmer(self, c1: str, c2: str) -> float:
        subsumer = self.lcs(c1, c2)
        if subsumer is None:
            return 0.0
        return 2 * self.depth(subsumer) / (self.depth(c1) + self.depth(c2))

    def wpath(self, c1: str, c2: str, k: float = 0.8) -> float:
        subsumer = self.lcs(c1, c2)
        if subsumer is None:
            return 0.0
        length = self.path_length(c1, c2)
        return 1.0 / (1.0 + length * (k ** self.information_content(subsumer)))

    # ---- recognition + aggregation ------------------------------------ #
    @lru_cache(maxsize=2048)
    def recognize(self, text: str | None) -> frozenset[str]:
        """Map free text to SEDO concept ids via surface keywords (NER)."""
        if not text:
            return frozenset()
        padded = f" {normalize_key(text)} "
        found: set[str] = set()
        for keyword, concept_id in self._keyword_index:
            if f" {keyword} " in padded:
                found.add(concept_id)
        return frozenset(found)

    def set_similarity(self, concepts_a, concepts_b, measure, weights=None) -> float | None:
        """Symmetric best-match average between two concept sets.

        Returns None when either side has no recognized concept, so the caller
        decides how to fall back (paper Sect. 5.5: unrecognized -> score drops).

        ``weights`` (optional) maps a concept id to a per-concept weight (e.g. corpus IDF), so
        rare, informative concepts count more than ubiquitous ones. None ⇒ every concept weighs 1
        (the plain average of the original paper).
        """
        a = list(concepts_a)
        b = list(concepts_b)
        if not a or not b:
            return None

        def weight(concept):
            return weights.get(concept, 1.0) if weights else 1.0

        def directed(source, target):
            numerator = sum(weight(s) * max(measure(s, t) for t in target) for s in source)
            denominator = sum(weight(s) for s in source)
            return numerator / denominator if denominator else 0.0

        return (directed(a, b) + directed(b, a)) / 2.0


@lru_cache(maxsize=1)
def get_sedo() -> Sedo:
    with open(_SEDO_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    return Sedo(data["concepts"])
