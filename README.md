# Thesis Dataset Import & Similarity API

FastAPI service để **import danh sách đề tài (thesis) từ Excel**, chuẩn hoá nội dung
(có hỗ trợ AI), chống trùng lặp, và tính **độ tương đồng** giữa các đề tài. Dùng
PostgreSQL + SQLAlchemy + Alembic.

## Mô hình dữ liệu

Mỗi đề tài (`thesis`) có đúng **5 trường nội dung**:

| Trường | Ý nghĩa |
|--------|---------|
| `title` | Tên đề tài |
| `description` | Mô tả |
| `scope` | Phạm vi |
| `objectives` | Mục tiêu |
| `expected_result` | Kết quả kỳ vọng |

Kèm `semester`, `program`, các phân loại nhiều-nhiều (domain, technology, semantic,
structure, lexical) và `content_hash` (băm 5 trường + semester + program để chống trùng).

## Chạy bằng Docker (khuyên dùng)

```bash
docker compose up -d --build
```

- `db`: PostgreSQL 15
- `app`: FastAPI — khi khởi động tự chạy `alembic upgrade head` rồi bật uvicorn.

Mở API docs (Swagger): **http://localhost:8000/docs** (truy cập `http://localhost:8000/`
sẽ tự chuyển hướng sang `/docs`).

### Biến môi trường (`.env`)

```env
DATABASE_URL=postgresql+psycopg://postgres:password@db:5432/fastapi_db
OPENAI_API_KEY=...        # tuỳ chọn — nếu có, AI sẽ suy ra trường còn thiếu
MAX_UPLOAD_MB=50
```

> `ALEMBIC_DATABASE_URL` (nếu đặt) dùng khi chạy Alembic từ **máy thật** (host).
> Chạy Alembic trong container thì ghi đè rỗng để dùng `DATABASE_URL`:
> `docker compose exec -e ALEMBIC_DATABASE_URL= app alembic upgrade head`

## Import Excel

`POST /api/v1/import/excel` (multipart, field `file`). Hỗ trợ `.xlsx` và `.xls`.

Bộ đọc tự dò dòng tiêu đề (quét 10 dòng đầu) và map cột theo tên. Các cột nhận diện:

| Trường | Tên cột chấp nhận (một trong số) |
|--------|-----------------------------------|
| **title** (bắt buộc) | `title`, `title en`, `english title`, `project title`, `tên đề tài` |
| description | `description`, `summary`, `mô tả` |
| scope | `scope`, `phạm vi` |
| objectives | `objective`, `objectives`, `mục tiêu` |
| expected_result | `expected result`, `kết quả`, `output` |
| semester | `semester`, `học kỳ` |
| program | `program`, `ngành`, `major` |
| technologies | `technology`, `technologies`, `tech stack`, `công nghệ` |
| domains | `domain`, `field`, `lĩnh vực` |

Dòng không có `title` sẽ bị bỏ qua. Nếu thiếu `description`/`scope`/`objectives`/
`expected_result` mà có `title`, hệ thống tự sinh (OpenAI nếu có key, ngược lại dùng
heuristic mặc định).

Ví dụ bằng dòng lệnh:

```bash
curl -X POST "http://localhost:8000/api/v1/import/excel" \
  -F "file=@sample_thesis_import.xlsx"
```

Kết quả thành công:

```json
{ "success": true, "message": "Import completed",
  "data": { "inserted": 3, "new_ids": [1, 2, 3], "errors": [] } }
```

## Chống trùng lặp

Dựa trên **cả 5 trường nội dung** (title, description, scope, objectives,
expected_result) + semester + program:

- **Trùng chính xác** → *bỏ qua* (không insert), ghi audit `SKIP_DUPLICATE`.
  So khớp qua `content_hash` (băm 5 trường + semester + program, chuẩn hoá
  hoa/thường & khoảng trắng). Ràng buộc DB `uq_thesis_content_hash` là lớp an toàn.
- **Gần trùng** (độ giống tổ hợp 5 trường **≥ 0.85**, cùng semester+program) → *vẫn
  insert* nhưng gắn cờ `needs_review = true`. Ngưỡng chỉnh tại
  `app/repositories/thesis_repository.py`.

## Các endpoint chính

| Method | Path | Chức năng |
|--------|------|-----------|
| POST | `/api/v1/import/excel` | Import Excel |
| GET | `/api/v1/theses` | Danh sách (phân trang + lọc) |
| GET | `/api/v1/theses/{id}` | Chi tiết + phân loại |
| GET | `/api/v1/theses/{id}/similarities` | Các cặp tương đồng |
| POST | `/api/v1/similarity/run-new` | Tính lại similarity thủ công |
| GET | `/health` | Health check |

## Chạy tests

```bash
docker compose exec app pytest -q
```

Test dùng SQLite trong bộ nhớ (không cần Postgres).

## Tool: học trọng số MDDM (`learn_mddm_weights.py`)

Script **độc lập** (không nằm trong app, không được cài vào image) dùng để **học trọng
số α/β/γ/δ** cho 4 kênh tương đồng (semantic, lexical, structure, domain) thay cho trọng
số gán cứng trong `app/utils/score_calculator.py` (`WEIGHTS`).

### Cài dependency (riêng, ngoài app)

```bash
pip install scikit-learn scipy pandas openpyxl
```

### Chạy thử với dữ liệu giả lập

```bash
python learn_mddm_weights.py --demo
```

### Chạy với dữ liệu thật

File CSV/XLSX gồm 4 cột điểm tương đồng (đã chuẩn hoá [0,1]) + 1 cột nhãn 0/1:

```
pair_id, s_sem, s_lex, s_str, s_dom, label
```

```bash
python learn_mddm_weights.py --input scored_pairs.xlsx \
  --col-sem s_sem --col-lex s_lex --col-str s_str --col-dom s_dom --col-label label
```

Script in ra bộ trọng số học được (Logistic Regression / Constrained optimize / Grid
search) và ngưỡng `tau`. Áp kết quả bằng cách cập nhật `WEIGHTS` trong
`app/utils/score_calculator.py`: `semantic=α, lexical=β, structure=γ, domain=δ`.
