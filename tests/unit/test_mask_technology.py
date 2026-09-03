"""Tests for technology masking in the semantic dimension (ICTA_REVIEW_ANALYSIS §H / 2B).

The expert ground truth ignores technology; stripping stack terms before the semantic encode stops
two same-stack topics from looking alike just because they share React/.NET/SQL Server.
"""

from app.utils import score_calculator as sc


def test_mask_removes_technology_keeps_business():
    masked = sc.mask_technology("Hotel booking system using React, ASP.NET Core and SQL Server")
    assert "react" not in masked
    assert "sql server" not in masked
    assert "asp net core" not in masked
    assert "booking" in masked and "hotel" in masked  # business words survive


def test_mask_empty_input():
    assert sc.mask_technology("") == ""
    assert sc.mask_technology(None) == ""


def test_mask_collapses_same_business_different_stack():
    # identical business meaning, different stacks → identical once the stack is masked away
    a = sc.mask_technology("Online food delivery platform built with React and PostgreSQL")
    b = sc.mask_technology("Online food delivery platform built with Angular and MongoDB")
    assert a == b
