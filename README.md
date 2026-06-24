# FastAPI Base

Minimal FastAPI project scaffold with PostgreSQL and Alembic.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Database

Set `DATABASE_URL` in `.env`.

```env
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/fastapi_db
```

## Run app

```bash
uvicorn app.main:app --reload
```

## Run PostgreSQL with Docker

```bash
docker compose up -d db
```

## Alembic migration

Create a migration:

```bash
alembic revision --autogenerate -m "init"
```

Apply migrations:

```bash
alembic upgrade head
```

## Endpoints

- `GET /`
- `GET /health`
- `GET /db/health`
- `GET /docs`
