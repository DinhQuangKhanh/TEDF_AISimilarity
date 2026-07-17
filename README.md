# TEDF — Thesis Topic Import & Similarity API

FastAPI service that **imports Software Engineering thesis topics from Excel**, normalizes
their content (optionally with AI), detects duplication, and scores **multi-dimensional
similarity** between topics. Backed by PostgreSQL + SQLAlchemy + Alembic.

This repository is the **data-import and schema component** of the DASSF study
(*Domain-Aware Multi-Dimensional Similarity for Detecting Structural Duplication in Software
Engineering Thesis Topics*): it ingests the raw topic corpus into a relational schema and
assembles the concatenated five-field representation used by the framework.

> **Scope note.** The similarity dimensions here are lightweight **token-based
> approximations**. The full DASSF method (Sentence-BERT embeddings and the SEDO ontology with
> Wu-Palmer / wpath measures) is **not implemented in this repository**. The MDDM fusion
> weights, the four-level decision scale, and the structural-duplication rule **do** follow the
> paper exactly.

## Topic model

Every thesis topic is a structured record of exactly **five content fields**:

| Field | Meaning |
|-------|---------|
| `title` | Short name of the topic |
| `description` | What the system does |
| `scope` | Included modules and technologies |
| `objectives` | Intended outcomes |
| `expected_result` | Concrete deliverable |

Plus `semester`, `program`, many-to-many classifications (domain, technology, semantic
category, structure type, lexical tag), and `content_hash` (a hash of the five fields +
semester + program, used for duplicate detection).

## Run with Docker (recommended)

```bash
docker compose up -d --build
```

- `db` — PostgreSQL 15
- `app` — FastAPI; runs `alembic upgrade head` on startup, then uvicorn

Open the API docs (Swagger UI): **http://localhost:8000/docs**
(visiting `http://localhost:8000/` redirects there).

### Environment variables (`.env`)

```env
DATABASE_URL=postgresql+psycopg://postgres:password@db:5432/fastapi_db
OPENAI_API_KEY=...        # optional - if set, missing fields are inferred by AI
MAX_UPLOAD_MB=50
```

> If `ALEMBIC_DATABASE_URL` is set, it is meant for running Alembic **from the host**. To run
> Alembic **inside the container**, override it so `DATABASE_URL` is used:
> ```bash
> docker compose exec -e ALEMBIC_DATABASE_URL= app alembic upgrade head
> ```

## Importing an Excel file

`POST /api/v1/import/excel` (multipart, field `file`). Supports `.xlsx` and `.xls`.

The reader auto-detects the header row (scanning the first 10 rows) and maps columns by name:

| Field | Accepted column names (any of) |
|-------|--------------------------------|
| **title** (required) | `title`, `title en`, `english title`, `project title`, `tên đề tài` |
| description | `description`, `summary`, `mô tả` |
| scope | `scope`, `phạm vi` |
| objectives | `objective`, `objectives`, `mục tiêu` |
| expected_result | `expected result`, `kết quả`, `output` |
| semester | `semester`, `học kỳ` |
| program | `program`, `ngành`, `major` |
| technologies | `technology`, `technologies`, `tech stack`, `công nghệ` |
| domains | `domain`, `field`, `lĩnh vực` |

Rows without a `title` are skipped. If `description`, `scope`, `objectives`, or
`expected_result` is missing but a title is present, the value is generated (via OpenAI when
`OPENAI_API_KEY` is set, otherwise a deterministic heuristic).

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/import/excel" \
  -F "file=@sample_thesis_import.xlsx"
```

Successful response:

```json
{ "success": true, "message": "Import completed",
  "data": { "inserted": 3, "new_ids": [1, 2, 3], "errors": [] } }
```

## Duplicate detection

Both checks read **all five content fields** plus `semester` and `program`:

- **Exact duplicate** → *skipped* (not inserted), audited as `SKIP_DUPLICATE`. Matched via
  `content_hash` (SHA-256 over the normalized five fields + semester + program, so casing and
  whitespace do not matter). The DB constraint `uq_thesis_content_hash` is the safety net.
- **Near duplicate** (token Jaccard over the combined five fields **≥ 0.85**, within the same
  semester and program) → *still inserted*, but flagged `needs_review = true`. The threshold
  lives in `app/repositories/thesis_repository.py`.

## Similarity scoring (MDDM)

Each pair of topics is scored on four dimensions and fused into a composite score:

```
S_composite = 0.30·S_sem + 0.20·S_lex + 0.30·S_str + 0.20·S_dom
```

| Dimension | Paper technique | Implemented here | Fields read |
|-----------|-----------------|------------------|-------------|
| Semantic | SBERT cosine | token Jaccard | all five fields |
| Lexical | TF-IDF Jaccard | **TF-IDF weighted Jaccard** (corpus IDF) | all five fields |
| Structural | Wu-Palmer over SEDO | token Jaccard + technology/structure tags | scope + description |
| Domain | wpath over SEDO | token Jaccard + domain tags | description + objectives |

### Four-level decision scale

| Level | Composite score | Recommended action |
|-------|-----------------|--------------------|
| Low | < 0.40 | Accept |
| Moderate | 0.40 – 0.65 | Warn; committee reviews |
| High | 0.65 – 0.85 | Require substantial revision |
| Critical | ≥ 0.85 | Reject |

### Structural duplication

A pair is a **structural duplication** (same tech stack, different business domain) when:

```
S_str >= TAU_STR  and  S_dom < TAU_DOM
```

Defaults are `TAU_STR = 0.65` and `TAU_DOM = 0.40` in `app/utils/score_calculator.py`. The
paper does not fix these numerically — tune them on a labeled set. The
`/api/v1/theses/{id}/similarities` endpoint returns `level`, `action`, and
`is_structural_duplication` for every pair, alongside the four per-dimension scores, so a
committee can see *why* a pair was flagged.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/import/excel` | Import topics from Excel |
| GET | `/api/v1/theses` | List topics (pagination + filters) |
| GET | `/api/v1/theses/{id}` | Topic detail with classifications |
| GET | `/api/v1/theses/{id}/similarities` | Similar pairs with scores and action |
| POST | `/api/v1/similarity/run-new` | Recompute similarity on demand |
| GET | `/health` | Health check |

## Running tests

```bash
docker compose exec app python -m pytest -q
```

Tests run against SQLite, so no PostgreSQL instance is required.

## Tool: learning the MDDM weights

`tools/learn_mddm_weights.py` is a **standalone** script (not imported by the app and not
installed into the Docker image) that learns the fusion weights α/β/γ/δ instead of hard-coding
them, using the procedure described in the paper: fit a logistic-regression model on the four
dimension scores, project its non-negative coefficients onto the simplex, and pick the
F1-maximizing decision threshold. It also cross-checks that result against a constrained
optimizer and a grid search.

### Install its dependencies (separate from the app)

```bash
pip install -r tools/requirements.txt
```

### Try it on simulated data

```bash
python tools/learn_mddm_weights.py --demo
```

### Run it on real scores

Provide a CSV/XLSX with the four dimension scores (normalized to `[0,1]`) and a binary label:

```
pair_id, s_sem, s_lex, s_str, s_dom, label
```

```bash
python tools/learn_mddm_weights.py --input scored_pairs.xlsx \
  --col-sem s_sem --col-lex s_lex --col-str s_str --col-dom s_dom --col-label label
```

To apply the learned weights, update `WEIGHTS` in `app/utils/score_calculator.py`
(`semantic=α, lexical=β, structure=γ, domain=δ`).
