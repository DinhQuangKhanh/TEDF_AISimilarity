"""Tests for the DB-free recent-capstone corpus loader (P4)."""

from app.services.corpus_loader import load_recent_capstone_corpus


def test_loads_both_semesters_full_content():
    corpus = load_recent_capstone_corpus()
    assert len(corpus) == 53                       # SP26 (40) + SU26 (13)
    assert {t.semester for t in corpus} == {"Spring 2026", "Summer 2026"}


def test_topics_carry_all_content_fields():
    corpus = load_recent_capstone_corpus()
    sample = corpus[0]
    assert sample.title
    assert sample.description and sample.scope and sample.objectives and sample.expected_result


def test_topics_have_tech_and_domain_tags():
    corpus = load_recent_capstone_corpus()
    # at least most topics recognise a tech stack; tags are objects exposing .name
    with_tech = [t for t in corpus if t.technologies]
    assert len(with_tech) >= 40
    assert all(hasattr(tag, "name") for t in with_tech for tag in t.technologies)
    # some topic maps onto a business domain via Module 1
    assert any(t.domains for t in corpus)
