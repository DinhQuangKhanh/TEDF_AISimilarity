from fastapi.testclient import TestClient

import app.models  # noqa: F401 - register every model with Base.metadata
from app.database import Base, engine
from app.main import app

client = TestClient(app)

PAYLOAD = {
    # thesis_id is required: it is the web system's project id, reused as the thesis id.
    "thesis_id": "11111111-1111-1111-1111-111111111111",
    "title": "Hotel Management System",
    "description": "A web system to manage hotel bookings and customers, built with React.",
    "scope": "React frontend, Node.js backend, PostgreSQL database, REST API.",
    "objectives": "Digitize hotel booking, manage rooms, generate reports.",
    "expected_result": "A deployed web application for hotel staff.",
    "semester": "2024-1",
    "program": "Software Engineering",
    "domains": ["Hotel"],
    "technologies": ["React", "Node.js", "PostgreSQL"],
}


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def _second_topic():
    second = dict(PAYLOAD)
    second.update(
        {
            "thesis_id": "22222222-2222-2222-2222-222222222222",
            "title": "Pharmacy Management System",
            "description": "A web system to manage pharmacy inventory and customers, built with React.",
            "objectives": "Digitize pharmacy sales, manage medicine stock, generate reports.",
            "expected_result": "A deployed web application for pharmacy staff.",
            "domains": ["Pharmacy"],
        }
    )
    return second


def test_create_returns_detail_with_all_five_fields():
    response = client.post("/api/v1/theses", json=PAYLOAD)
    assert response.status_code == 201
    data = response.json()["data"]
    thesis = data["thesis"]
    assert thesis["title"] == PAYLOAD["title"]
    assert thesis["description"] == PAYLOAD["description"]
    assert thesis["scope"] == PAYLOAD["scope"]
    assert thesis["objectives"] == PAYLOAD["objectives"]
    assert thesis["expected_result"] == PAYLOAD["expected_result"]
    # Nothing else in the corpus yet.
    assert data["similarities"] == []


def test_create_returns_similarity_against_existing_topics():
    client.post("/api/v1/theses", json=PAYLOAD)
    response = client.post("/api/v1/theses", json=_second_topic())
    assert response.status_code == 201

    items = response.json()["data"]["similarities"]
    assert len(items) == 1
    item = items[0]
    for key in ("semantic_score", "lexical_score", "structure_score", "domain_score", "overall_score"):
        assert 0.0 <= item[key] <= 1.0
    assert item["level"] in {"Low", "Moderate", "High", "Critical"}
    assert item["action"]
    # Same tech stack, different business domain -> the paper's structural class.
    assert item["structure_score"] > item["domain_score"]
    assert item["is_structural_duplication"] is True


def test_exact_duplicate_is_rejected():
    assert client.post("/api/v1/theses", json=PAYLOAD).status_code == 201
    assert client.post("/api/v1/theses", json=PAYLOAD).status_code == 409


def test_missing_title_is_rejected():
    payload = dict(PAYLOAD)
    payload["title"] = ""
    assert client.post("/api/v1/theses", json=payload).status_code == 422
