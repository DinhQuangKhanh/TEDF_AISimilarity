#!/usr/bin/env python3
"""Xem AI XỬ LÝ TRÙNG LẶP một đề tài — từng bước, input/output rõ ràng.

Chạy đúng pipeline mà endpoint ``POST /api/v1/similarity/analyze`` và trang ``/demo`` dùng
(``app.services.analyze_service.analyze_topic``), nhưng in ra terminal thay vì trả JSON — nên
những gì bạn đọc ở đây CHÍNH LÀ những gì hệ thống thật sự làm, không phải bản mô phỏng.

    BƯỚC 1  Nhận đề tài          →  trường nào đã điền, chế độ so khớp (đủ nội dung / chỉ tiêu đề)
    BƯỚC 2  Tiền xử lý           →  token đã bỏ stop-word, đưa về gốc, gộp đồng nghĩa
    BƯỚC 3  Nhận diện SEDO       →  (công nghệ, phương pháp, lĩnh vực, loại tác vụ)
    BƯỚC 4  Encoder ngữ nghĩa    →  backend đang chạy (MiniLM thật hay fallback)
    BƯỚC 5  Nạp kho + mô hình    →  N đề tài cũ, TF-IDF, IDF khái niệm
    BƯỚC 6  Chấm 4 chiều         →  bảng điểm với từng đề tài cũ
    BƯỚC 7  Gộp điểm MDDM        →  phép tính α·S_sem + β·S_lex + γ·S_str + δ·S_dom = S
    BƯỚC 8  Phân mức & xử lý     →  Low / Moderate / High / Critical + khuyến nghị

Chạy (trong backend/src, KHÔNG cần database):

    python tools/trace_pipeline.py --example
    python tools/trace_pipeline.py --title "Hotel Management System using React and Node.js"
    python tools/trace_pipeline.py --title "..." --description "..." --tech "React, Node.js"
    python tools/trace_pipeline.py --example --json out.json     # kèm nhật ký máy đọc được
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _trace_console import banner, bar, head, io, note, rule, status_mark, wrap  # noqa: E402
from app.services.analyze_service import analyze_topic  # noqa: E402
from app.services.corpus_loader import load_recent_capstone_corpus  # noqa: E402
from app.services.lexical import build_lexical_scorer  # noqa: E402
from app.services.semantic_encoder import backend_name  # noqa: E402

_LEVEL_NOTE = {
    "Low": "khác biệt đủ rõ",
    "Moderate": "cần hội đồng xem lại",
    "High": "phải chỉnh sửa đáng kể",
    "Critical": "từ chối",
}
_DIM_LABEL = {"semantic": "Ngữ nghĩa", "lexical": "Từ vựng",
              "structure": "Cấu trúc", "domain": "Lĩnh vực"}


class Tag:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class Topic:
    """Đề tài người dùng nhập — cùng hình dạng với đối tượng endpoint /analyze dựng lên."""

    def __init__(self, title, description=None, scope=None, objectives=None,
                 expected_result=None, technologies=()):
        self.title = title
        self.description = description
        self.scope = scope
        self.objectives = objectives
        self.expected_result = expected_result
        self.technologies = [Tag(t) for t in technologies]
        self.structures = []
        self.domains = []


# ── in một giá trị input/output cho gọn mắt ─────────────────────────────────────────
def _fmt(value, limit: int = 12) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "có" if value else "không"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        if not value:
            return "— (rỗng)"
        if all(isinstance(v, dict) for v in value):
            return f"{len(value)} mục"
        shown = ", ".join(str(v) for v in value[:limit])
        return shown + (f" … (+{len(value) - limit})" if len(value) > limit else "")
    if isinstance(value, dict):
        return "  ".join(f"{k}={_fmt(v, 6)}" for k, v in value.items())
    text = str(value)
    return text if len(text) <= 150 else text[:147] + "…"


def _block(label: str, payload) -> None:
    """In một khối input/output: mỗi khoá một dòng, giá trị dài thì tự ngắt."""
    if not payload:
        return
    if isinstance(payload, str):
        io(label, *wrap(payload, 74))
        return
    if isinstance(payload, list):
        io(label, *[str(x) for x in payload])
        return
    lines = []
    for key, value in payload.items():
        rendered = _fmt(value)
        wrapped = wrap(rendered, 60)
        lines.append(f"{key:<18} = {wrapped[0].strip()}")
        lines.extend(" " * 21 + w.strip() for w in wrapped[1:])
    io(label, *lines)


def render_step(index: int, total: int, step: dict) -> None:
    head(index, total, f"{step['name']}   [{status_mark(step['status'])} {step['status']}]")
    if step.get("detail"):
        io("Ý NGHĨA", *wrap(step["detail"], 74))
    _block("INPUT", step.get("input"))
    formula = step.get("formula")
    if formula:
        _block("CÔNG THỨC", formula if isinstance(formula, (list, str)) else str(formula))
    _block("OUTPUT", step.get("output"))

    # Bảng điểm chi tiết của bước chấm 4 chiều.
    if step["id"] == "scoring" and (step.get("output") or {}).get("rows"):
        print(f"\n    {'S_sem':>6}{'S_lex':>7}{'S_str':>7}{'S_dom':>7}{'GỘP':>8}   đề tài trong kho")
        rule()
        for row in step["output"]["rows"]:
            print(f"    {row['semantic']:>6.2f}{row['lexical']:>7.2f}{row['structure']:>7.2f}"
                  f"{row['domain']:>7.2f}{row['overall']:>8.3f}   {(row['title'] or '')[:44]}")

    # Phép tính gộp điểm, viết ra từng số hạng.
    if step["id"] == "mddm" and (step.get("output") or {}).get("terms"):
        print()
        for term in step["output"]["terms"]:
            print(f"    {term['symbol']} · {_DIM_LABEL[term['dim']]:<10}"
                  f"{term['weight']:>6.2f} × {term['score']:>5.2f} = {term['product']:>6.4f}   "
                  f"{bar(term['product'], 24, 0, 0.6)}")
        rule()
        print(f"    {'':<14}{'TỔNG':>15} = {step['output']['overall']:>6.4f}")


def render_matches(result: dict, top_n: int) -> None:
    matches = result.get("topMatches") or []
    if not matches:
        return
    print("\n" + "═" * 92)
    print(f" KẾT QUẢ · {min(top_n, len(matches))} đề tài giống nhất")
    print("═" * 92)
    for rank, m in enumerate(matches[:top_n], 1):
        b = m["breakdown"]
        flag = "  ⚠ TRÙNG CẤU TRÚC (cùng stack, khác lĩnh vực)" if m["is_structural_duplication"] else ""
        print(f"\n  #{rank}  {m['overall_score'] * 100:5.1f}%  [{m['level']}] "
              f"{m['action']} — {_LEVEL_NOTE.get(m['level'], '')}{flag}")
        print(f"      ↔ {(m.get('otherTitle') or '')[:78]}"
              + (f"   ({m['otherSemester']})" if m.get("otherSemester") else ""))
        for dim in ("semantic", "lexical", "structure", "domain"):
            print(f"        {_DIM_LABEL[dim]:<10} {bar(b[dim], 34)} {b[dim] * 100:5.1f}%")
        shared = m.get("shared_concepts") or {}
        for key, label in (("tech", "công nghệ chung"), ("domain", "lĩnh vực chung"),
                           ("task", "tác vụ chung")):
            if shared.get(key):
                print(f"        {label:<16} {', '.join(shared[key])}")
        if m.get("reasons"):
            print(f"        lý do            {'; '.join(m['reasons'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace từng bước luồng AI kiểm tra trùng lặp.")
    ap.add_argument("--title", help="tiêu đề đề tài cần kiểm tra")
    ap.add_argument("--description", default=None)
    ap.add_argument("--scope", default=None)
    ap.add_argument("--objectives", default=None)
    ap.add_argument("--expected", dest="expected_result", default=None)
    ap.add_argument("--tech", default="", help="công nghệ, phân tách bằng dấu phẩy")
    ap.add_argument("--top", type=int, default=5, help="số đề tài giống nhất hiển thị")
    ap.add_argument("--example", action="store_true", help="dùng đề tài ví dụ có sẵn")
    ap.add_argument("--json", dest="json_out", help="ghi nhật ký đầy đủ ra file JSON")
    args = ap.parse_args()

    if args.example and not args.title:
        args.title = "Hotel Management System using React and Node.js"
        args.description = args.description or ("A web system to manage hotel bookings, rooms, "
                                                "customers and staff shifts.")
        args.scope = args.scope or "Booking, room inventory, invoicing and customer profiles."
        args.tech = args.tech or "React, Node.js, PostgreSQL"
    if not args.title:
        ap.error("cần --title (hoặc --example)")

    banner(" AI KIỂM TRA TRÙNG LẶP ĐỀ TÀI — LUỒNG XỬ LÝ TỪNG BƯỚC ",
           f" {args.title[:80]} ")

    topic = Topic(args.title, args.description, args.scope, args.objectives,
                  args.expected_result, [t.strip() for t in args.tech.split(",") if t.strip()])

    t0 = time.perf_counter()
    corpus = load_recent_capstone_corpus()
    lexical_model = build_lexical_scorer(corpus) if corpus else None
    result = analyze_topic(topic, corpus, lexical_model, top_k=max(args.top, 5),
                           semantic_backend=backend_name())
    elapsed = time.perf_counter() - t0

    steps = result["steps"]
    for index, step in enumerate(steps, 1):
        render_step(index, len(steps), step)

    render_matches(result, args.top)

    print("\n" + "─" * 92)
    weights = result["weights"]
    print(f"  Trọng số đang dùng: α={weights['semantic']:.2f} β={weights['lexical']:.2f} "
          f"γ={weights['structure']:.2f} δ={weights['domain']:.2f}   "
          f"(nguồn: {result['weightsSource']})")
    print(f"  Kho đối chiếu: {result['corpusSize']} đề tài · tổng thời gian {elapsed:.2f}s")
    print("  Xem 4 trọng số này được HỌC ra sao:  python tools/trace_tuning.py")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"  → nhật ký JSON: {args.json_out}")


if __name__ == "__main__":
    main()
