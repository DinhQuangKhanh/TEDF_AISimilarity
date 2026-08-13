# TEDF — Thesis Topic Import & Similarity API

FastAPI service that **imports Software Engineering thesis topics from Excel**, normalizes
their content (optionally with AI), detects duplication, and scores **multi-dimensional
similarity** between topics. Backed by PostgreSQL + SQLAlchemy + Alembic.

This repository is the **data-import and schema component** of the DASSF study
(*Domain-Aware Multi-Dimensional Similarity for Detecting Structural Duplication in Software
Engineering Thesis Topics*): it ingests the raw topic corpus into a relational schema and
assembles the concatenated five-field representation used by the framework.

> **Scope note.** The **structural** and **domain** dimensions are grounded in the SEDO
> ontology with **Wu-Palmer / wpath** measures, as in the paper. The **semantic** dimension
> here is a token-Jaccard approximation (the paper uses Sentence-BERT). The MDDM fusion weights,
> the four-level decision scale, and the structural-duplication rule follow the paper exactly.

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

## Submitting a single topic

`POST /api/v1/theses` adds one topic the same way the Excel importer does — normalize (AI fills
any missing field), map to the schema, run both duplicate checks, save, then score it against
every existing topic. The similarity results come back **in the same response**, so no second
call is needed.

```bash
curl -X POST http://localhost:8000/api/v1/theses \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Pharmacy Management System",
    "description": "A web system to manage pharmacy inventory and customers, built with React.",
    "scope": "React frontend, Node.js backend, PostgreSQL database, REST API.",
    "objectives": "Digitize pharmacy sales, manage medicine stock, generate reports.",
    "expected_result": "A deployed web application for pharmacy staff.",
    "semester": "2024-1",
    "program": "Software Engineering",
    "domains": ["Pharmacy"],
    "technologies": ["React", "Node.js", "PostgreSQL"]
  }'
```

```json
{
  "success": true, "message": "Thesis created",
  "data": {
    "thesis": { "thesis_id": 2, "title": "Pharmacy Management System", "...": "..." },
    "needs_review": false,
    "similarities": [
      { "thesis_a_id": 1, "thesis_b_id": 2,
        "semantic_score": 0.7353, "lexical_score": 0.664,
        "structure_score": 1.0, "domain_score": 0.2052,
        "overall_score": 0.6944, "level": "High",
        "action": "Require substantial revision",
        "is_structural_duplication": true,
        "reason": ["same tech stack with a different business domain", "..."] }
    ]
  }
}
```

Responses: `201` on success, `409` when an identical topic already exists, `422` when the title
is missing. `POST /api/v1/import/excel` and `POST /api/v1/similarity/run-new` also include a
`similarities` array in their response.

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
| Structural | Wu-Palmer over SEDO | **Wu-Palmer over SEDO** | scope + description (+ tech/structure tags) |
| Domain | wpath over SEDO | **wpath over SEDO** | description + objectives (+ domain tags) |

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

## SEDO ontology

The structural and domain dimensions are grounded in **SEDO** (Software Engineering Domain
Ontology), a four-layer hierarchy (Technical Stack, Methodology, Domain Entity, Task Type) with
74 concepts in 18 parent classes and 238 surface keywords. It lives in
[`app/ontology/sedo.json`](app/ontology/sedo.json) and is loaded by `app/ontology/sedo.py`.

- **Wu-Palmer** (structural): `sim = 2·depth(LCS) / (depth(c1) + depth(c2))` — e.g.
  `wu_palmer(React, Angular) = 0.75` (interchangeable Frontend frameworks).
- **wpath** (domain): `sim = 1 / (1 + path(c1,c2)·k^IC(LCS))` with intrinsic IC — e.g.
  `wpath(Hotel, Pharmacy) ≈ 0.205` (different business domains).
- **Recognition (NER):** free text is mapped to concepts by matching the 238 surface keywords.

Configuration (`.env`):

```env
SEDO_FALLBACK_TOKENS=false   # default: matches the paper (unrecognized concept -> score drops)
WPATH_K=0.8                  # wpath k parameter
```

When SEDO recognizes no concept for a dimension, the default (paper) behavior is to let the
score drop — reproducing the false negatives discussed in the paper's error analysis (Sect 5.5,
e.g. "Microsoft SQL" is not in the vocabulary). Set `SEDO_FALLBACK_TOKENS=true` to fall back to
token Jaccard instead.

## Reproducibility — verifying the formulas

The paper's numeric claims (Wu-Palmer, wpath, MDDM weights, the four-level scale, and the
structural-duplication rule) are pinned as machine-checkable cases in
[`tests/fixtures/paper_formulas.json`](tests/fixtures/paper_formulas.json). Each case lists its
inputs, the expected value, and the paper location it comes from.

Run them through the real code:

```bash
# as part of the test suite
docker compose exec app python -m pytest tests/test_paper_formulas.py -v

# or as a standalone PASS/FAIL report (no pytest)
docker compose exec app python tools/verify_formulas.py
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/import/excel` | Import topics from Excel |
| POST | `/api/v1/theses` | **Add one new topic** and score it against the corpus |
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


Đề tài của hệ thống web đang có projectId kiểu uuid
similarity lại nhận list[int] ids làm request_body
---> Sửa lại thành kiểu list[uuid]