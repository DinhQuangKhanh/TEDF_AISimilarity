import io
import os

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///./test_api.sqlite"

from app.database import Base
from app.main import app

engine = create_engine("sqlite:///./test_api.sqlite", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

client = TestClient(app)


def make_excel_bytes():
    df = pd.DataFrame(
        {
            "title_en": ["Project Alpha"],
            "title_vn": ["Dự án Alpha"],
            "description": ["A test project"],
            "semester": ["2025 Spring"],
            "program": ["Software Engineering"],
            "technologies": ["React, FastAPI, Postgres"],
            "domains": ["E-commerce"],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def test_import_success_and_duplicate_skip():
    excel_bytes = make_excel_bytes()
    response = client.post(
        "/api/v1/import/excel",
        files={"file": ("test.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["inserted"] == 1

    duplicate_response = client.post(
        "/api/v1/import/excel",
        files={"file": ("test.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["data"]["inserted"] == 0
