"""Diagnostic: score the user's 'EcoTrack' topic against the seeded corpus and print the top-5,
so the effect of P3 (no TaskType in structural + concept-IDF weighting) is visible on the exact
case that motivated it. Toggle P3 via env to compare:

    # function-centric structural (default)
    python tools/measure_ecotrack.py
    # old tech-centric structural
    STRUCT_LAYERS="TechnicalStack,Methodology" CONCEPT_IDF_WEIGHTING=false python tools/measure_ecotrack.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import config  # noqa: E402
from app.services.corpus_loader import load_recent_capstone_corpus  # noqa: E402
from app.services.lexical import build_lexical_scorer  # noqa: E402
from app.utils import score_calculator as sc  # noqa: E402
from evaluate import EvalThesis  # noqa: E402

ECOTRACK = EvalThesis(
    title="EcoTrack – Smart Household Waste Management System using React and ASP.NET Core",
    description=("EcoTrack is a smart household waste management platform that helps users track, "
                 "classify and manage daily waste, with collection scheduling, recycling tracking "
                 "and environmental impact statistics, plus an admin portal."),
    scope="User management, waste classification, collection management, recycling tracking, statistics, reporting.",
    objectives="Centralized platform for household waste; classify waste; schedule and track collection; encourage recycling.",
    expected_result="A web-based waste management platform with user and admin interfaces.",
    technologies=["React", "TypeScript", "ASP.NET Core", "C#", "Entity Framework Core", "PostgreSQL",
                  "Redis", "Docker", "REST API", "JWT", "Tailwind CSS"],
)


def main() -> None:
    corpus = load_recent_capstone_corpus()   # Spring 2026 + Summer 2026, full content
    lexical = build_lexical_scorer(corpus)
    concept_idf = sc.build_concept_idf(corpus)

    scored = sorted(
        ((cand, sc.calculate_scores(ECOTRACK, cand, lexical, concept_idf)) for cand in corpus),
        key=lambda pair: pair[1]["overall_score"], reverse=True,
    )[:5]

    print(f"flags: STRUCT_LAYERS={sorted(config.STRUCT_LAYERS)} "
          f"CONCEPT_IDF_WEIGHTING={config.CONCEPT_IDF_WEIGHTING}")
    print(f"{'ovr':>5}{'sem':>6}{'lex':>6}{'str':>6}{'dom':>6}{'  struct-dup':>12}  title")
    for cand, s in scored:
        print(f"{s['overall_score']*100:>4.0f}%{s['semantic_score']*100:>5.0f}%{s['lexical_score']*100:>5.0f}%"
              f"{s['structure_score']*100:>5.0f}%{s['domain_score']*100:>5.0f}%"
              f"{('YES' if s['structural_duplication'] else 'no'):>12}  {(cand.title or '')[:64]}")


if __name__ == "__main__":
    main()
