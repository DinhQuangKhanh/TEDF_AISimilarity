#!/usr/bin/env python3
"""Xem AI HỌC bộ 4 trọng số (α, β, γ, δ) bằng grid search — từng bước, input/output rõ ràng.

`tools/tune_weights.py` chạy grid search rồi chỉ in ra kết quả cuối. Script này chạy **cùng một**
grid search (gọi lại ``evaluate.tune_weights``, không cài đặt lại thuật toán) nhưng gắn một
``observer`` vào vòng lặp, nên mỗi ứng viên được thử đều có thể quan sát được:

    BƯỚC 1  Nạp dữ liệu học        →  corpus đề tài cũ + benchmark gán nhãn của giảng viên
    BƯỚC 2  Trích đặc trưng        →  ma trận điểm 4 chiều cho mọi cặp (query × corpus)
    BƯỚC 3  Gán nhãn vàng          →  duplicationResult của giảng viên → mức Low/Moderate/High/Critical
    BƯỚC 4  Dựng không gian tìm kiếm →  mọi (α,β,γ,δ) ≥ 0, tổng = 1, bước nhảy `--step`
    BƯỚC 5  Quét lưới              →  mỗi ứng viên: gộp điểm → phân mức → macro-F1 (in mỗi lần phá kỷ lục)
    BƯỚC 6  Bảng xếp hạng          →  top ứng viên tốt nhất
    BƯỚC 7  Phân tích biên         →  macro-F1 trung bình theo từng giá trị của từng trọng số → vì sao thắng
    BƯỚC 8  Chốt & so sánh         →  trọng số học được vs trọng số mặc định của bài báo

Chạy (trong backend/src, KHÔNG cần database):

    python tools/trace_tuning.py                  # bước nhảy 0.05 → 1771 ứng viên
    python tools/trace_tuning.py --step 0.1       # thô hơn, chạy nhanh, dễ đọc log
    python tools/trace_tuning.py --save-weights   # ghi đè luôn data/tuned_weights.json

Ngoài log ra màn hình, script ghi ``data/tuning_trace.json`` để trang /demo vẽ lại quá trình học.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _trace_console import WIDTH as _W  # noqa: E402
from _trace_console import banner, bar, head, io, note, rule, wtuple  # noqa: E402
from app.utils import score_calculator as sc  # noqa: E402
from evaluate import (  # noqa: E402
    _DEFAULT_WEIGHTS, _DIMS, level_metrics, prepare_benchmark, simplex_size, tune_weights,
)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_TRACE_FILE = os.path.join(_DATA_DIR, "tuning_trace.json")
_WEIGHTS_FILE = os.path.join(_DATA_DIR, "tuned_weights.json")
_GREEK = {"semantic": "α", "lexical": "β", "structure": "γ", "domain": "δ"}


# ── các bước ────────────────────────────────────────────────────────────────────────
def step1_load(total: int):
    head(1, total, "Nạp dữ liệu học (corpus đề tài cũ + benchmark của giảng viên)")
    io("INPUT",
       "seed_thesis.py            → đề tài các kỳ trước (Fall-25 / Spring-26 / Summer-26)",
       "data/FA26_topics_deduplicated.json → mỗi đề tài Fall-26 giảng viên đã ghi:",
       "     • topicMostLikelyDuplicate = đề tài cũ bị cho là trùng nhất",
       "     • duplicationResult        = điểm trùng lặp giảng viên chấm (0…1)")
    t0 = time.perf_counter()
    queries, corpus, matrix = prepare_benchmark()
    elapsed = time.perf_counter() - t0

    resolved = [q for q in queries if q["target_idx"] is not None]
    labelled = [q for q in resolved if q["gold_score"] is not None]
    io("OUTPUT",
       f"corpus            = {len(corpus)} đề tài cũ",
       f"query (Fall-26)   = {len(queries)} đề tài cần kiểm tra",
       f"khớp được về corpus = {len(resolved)}/{len(queries)}  (Jaccard tiêu đề ≥ 0.5)",
       f"có nhãn điểm       = {len(labelled)} cặp → đây là tập dùng để học trọng số",
       f"thời gian          = {elapsed:.1f}s (mã hoá MiniLM + chấm điểm — CHỈ chạy 1 lần)")
    note("Lưu ý: bước tốn kém là mã hoá ngữ nghĩa, KHÔNG phải grid search.")
    return queries, corpus, matrix, labelled


def step2_matrix(total: int, queries, corpus, matrix, labelled):
    head(2, total, "Trích đặc trưng — chấm 4 chiều cho mọi cặp (query × corpus)")
    io("INPUT", f"{len(queries)} query × {len(corpus)} đề tài cũ = {len(queries) * len(corpus):,} cặp")
    io("CÔNG THỨC",
       "S_sem = cosine(MiniLM(A), MiniLM(B))                     — nghĩa",
       "S_lex = cosine(TF-IDF(A), TF-IDF(B))                     — từ vựng",
       "S_str = Wu-Palmer trên SEDO {TechnicalStack ∪ Methodology} — công nghệ/kiến trúc",
       "S_dom = wpath(k=0.8) trên SEDO {DomainEntity}             — lĩnh vực nghiệp vụ")
    io("OUTPUT",
       f"ma trận {len(queries)}×{len(corpus)}, mỗi ô = (S_sem, S_lex, S_str, S_dom)",
       "4 con số này CỐ ĐỊNH trong suốt quá trình học — grid search chỉ đổi trọng số gộp.")

    print(f"\n  Mẫu {min(6, len(labelled))} cặp (query ↔ đề tài giảng viên cho là trùng):")
    print(f"    {'S_sem':>6}{'S_lex':>7}{'S_str':>7}{'S_dom':>7}{'gold':>7}   đề tài")
    rule()
    samples = []
    for q in labelled[:6]:
        dims = matrix[q["_row"]][q["target_idx"]]
        title = (q["title"] or "")[:44]
        print(f"    {dims[0]:>6.2f}{dims[1]:>7.2f}{dims[2]:>7.2f}{dims[3]:>7.2f}"
              f"{float(q['gold_score']):>7.2f}   {title}")
        samples.append({"query": q["title"], "target": q["gold_target"],
                        "semantic": round(dims[0], 4), "lexical": round(dims[1], 4),
                        "structure": round(dims[2], 4), "domain": round(dims[3], 4),
                        "gold": round(float(q["gold_score"]), 4)})
    return samples


def step3_labels(total: int, labelled):
    head(3, total, "Gán nhãn vàng — điểm của giảng viên → mức trùng lặp (Table III)")
    io("INPUT", f"duplicationResult ∈ [0,1] của {len(labelled)} cặp có nhãn")
    io("CÔNG THỨC", "  ".join(f"≥{lo:.2f} → {lv}" for lo, lv in sc.LEVEL_THRESHOLDS))
    gold_levels = [sc.level_for(float(q["gold_score"])) for q in labelled]
    dist = {lv: gold_levels.count(lv) for _, lv in sc.LEVEL_THRESHOLDS if gold_levels.count(lv)}
    io("OUTPUT", "phân bố nhãn: " + "  ".join(f"{k}={v}" for k, v in dist.items()))
    note("Mục tiêu học: tìm (α,β,γ,δ) sao cho mức AI đoán khớp cột nhãn này nhất.")
    return dist


def step4_space(total: int, step: float):
    n = int(round(1 / step))
    size = simplex_size(n)
    head(4, total, "Dựng không gian tìm kiếm — lưới trên simplex")
    io("INPUT", f"bước nhảy = {step}  →  mỗi trọng số nhận giá trị 0, {step}, {2 * step:.2f}, …, 1.00")
    io("RÀNG BUỘC", "α, β, γ, δ ≥ 0   và   α + β + γ + δ = 1")
    io("CÔNG THỨC", f"số ứng viên = C(n+3, 3) với n = 1/bước = {n}  →  {size:,}")
    preview = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, step, 1 - step), (0.25, 0.25, 0.25, 0.25)]
    io("OUTPUT",
       f"{size:,} bộ trọng số ứng viên, ví dụ: " + " ".join(wtuple(w) for w in preview),
       "Vét cạn: không có gradient, không có ngẫu nhiên → chạy lại luôn ra cùng kết quả.")
    return n, size


def step5_scan(total: int, queries, matrix, n: int, size: int, max_logs: int):
    head(5, total, f"Quét lưới — thử lần lượt {size:,} ứng viên")
    io("INPUT", f"ma trận điểm 4 chiều (cố định) + {size:,} bộ trọng số ứng viên")
    io("MỖI Ư.VIÊN",
       "1) gộp điểm  : S = α·S_sem + β·S_lex + γ·S_str + δ·S_dom  cho từng cặp",
       "2) phân mức  : S → Low / Moderate / High / Critical (ngưỡng Table III)",
       "3) so nhãn   : macro-F1 giữa mức AI đoán và mức của giảng viên")
    io("CHỌN", "macro-F1 cao nhất; hoà thì xét AUC, rồi tới độ chính xác mức")
    print(f"\n  Nhật ký — chỉ in khi có ứng viên PHÁ KỶ LỤC (tốt hơn mọi ứng viên trước đó):")
    print(f"    {'#':>6}  {'(α, β, γ, δ)':<26}{'macroF1':>9}{'AUC':>7}{'acc':>7}   tiến bộ")
    rule()

    improvements: list[dict] = []
    all_points: list[tuple] = []
    state = {"logged": 0, "prev": 0.0}

    def observer(i, w, m, improved):
        all_points.append((w, m["macro_f1"], m["auc"] or 0.0, m["level_accuracy"]))
        if not improved:
            return
        improvements.append({"i": i, "weights": [round(x, 4) for x in w],
                             "macroF1": m["macro_f1"], "auc": m["auc"],
                             "accuracy": m["level_accuracy"]})
        if state["logged"] < max_logs:
            delta = m["macro_f1"] - state["prev"]
            print(f"    {i:>6}  {wtuple(w):<26}{m['macro_f1']:>9.4f}{(m['auc'] or 0):>7.3f}"
                  f"{m['level_accuracy']:>7.3f}   {bar(m['macro_f1'], 18)} {delta:+.4f}")
            state["logged"] += 1
        state["prev"] = m["macro_f1"]

    t0 = time.perf_counter()
    best = tune_weights(queries, matrix, n, observer=observer)
    elapsed = time.perf_counter() - t0

    if len(improvements) > state["logged"]:
        print(f"    … và {len(improvements) - state['logged']} lần cải thiện nữa "
              f"(dùng --max-logs để in thêm)")
    io("OUTPUT",
       f"đã thử       = {len(all_points):,} ứng viên trong {elapsed:.3f}s "
       f"({len(all_points) / max(elapsed, 1e-9):,.0f} ứng viên/giây)",
       f"số lần cải thiện = {len(improvements)}",
       f"quán quân    = {wtuple(best)}")
    return best, improvements, all_points, elapsed


def step6_leaderboard(total: int, all_points, top_k: int):
    head(6, total, f"Bảng xếp hạng — top {top_k} bộ trọng số")
    io("INPUT", f"toàn bộ {len(all_points):,} ứng viên đã chấm ở bước 5")
    ranked = sorted(all_points, key=lambda p: (p[1], p[2], p[3]), reverse=True)[:top_k]
    print(f"\n    {'#':>3}  {'α_sem':>6}{'β_lex':>7}{'γ_str':>7}{'δ_dom':>7}"
          f"{'macroF1':>10}{'AUC':>7}{'acc':>7}")
    rule()
    board = []
    for rank, (w, f1, auc, acc) in enumerate(ranked, 1):
        mark = " ←" if rank == 1 else ""
        print(f"    {rank:>3}  {w[0]:>6.2f}{w[1]:>7.2f}{w[2]:>7.2f}{w[3]:>7.2f}"
              f"{f1:>10.4f}{auc:>7.3f}{acc:>7.3f}{mark}")
        board.append({"rank": rank, "weights": [round(x, 4) for x in w],
                      "macroF1": round(f1, 4), "auc": round(auc, 4), "accuracy": round(acc, 4)})
    io("OUTPUT", "top đầu bảng thường cùng một 'vùng' trọng số → kết quả ổn định, không phải may rủi")
    return board


def step7_marginals(total: int, all_points, step: float):
    head(7, total, "Phân tích biên — mỗi trọng số đóng góp thế nào?")
    io("INPUT", f"{len(all_points):,} ứng viên, nhóm theo giá trị của từng trọng số")
    io("CÔNG THỨC", "với mỗi chiều d và mỗi giá trị v: trung bình macro-F1 của mọi ứng viên có w[d] = v")
    marginals = {}
    for idx, dim in enumerate(_DIMS):
        buckets: dict[float, list[float]] = {}
        for w, f1, _auc, _acc in all_points:
            buckets.setdefault(round(w[idx], 4), []).append(f1)
        rows = [(v, sum(f1s) / len(f1s), len(f1s)) for v, f1s in sorted(buckets.items())]
        marginals[dim] = [{"value": v, "meanMacroF1": round(m, 4), "n": k} for v, m, k in rows]

        lo = min(m for _, m, _ in rows)
        hi = max(m for _, m, _ in rows)
        best_v = max(rows, key=lambda r: r[1])[0]
        trend = "↑ tăng trọng số → tốt hơn" if rows[-1][1] > rows[0][1] else "↓ tăng trọng số → tệ hơn"
        print(f"\n    {_GREEK[dim]} · {dim:<10} (tốt nhất quanh {best_v:.2f})   {trend}")
        for v, m, k in rows[:: max(1, len(rows) // 10)]:
            print(f"      w={v:.2f}  {bar(m, 30, lo, hi)}  macro-F1 {m:.4f}   (n={k})")
    io("OUTPUT", "chiều nào có đường dốc lên → mang tín hiệu thật; đường phẳng/dốc xuống → gây nhiễu",
       "n = số ứng viên trong nhóm; w lớn có ÍT ứng viên nên trung bình ở đuôi kém tin cậy hơn")
    return marginals


def step8_final(total: int, queries, matrix, best, step, n, size, elapsed):
    head(total, total, "Chốt kết quả — trọng số học được vs mặc định của bài báo")
    best_m = level_metrics(queries, matrix, best)
    default_m = level_metrics(queries, matrix, _DEFAULT_WEIGHTS)
    io("INPUT", f"quán quân {wtuple(best)}  vs  mặc định {wtuple(_DEFAULT_WEIGHTS)}")
    print(f"\n    {'':<12}{'α_sem':>6}{'β_lex':>7}{'γ_str':>7}{'δ_dom':>7}"
          f"{'macroF1':>10}{'acc':>7}{'adj':>7}{'AUC':>7}{'binF1':>8}")
    rule()
    for label, w, m in (("bài báo", _DEFAULT_WEIGHTS, default_m), ("học được", best, best_m)):
        print(f"    {label:<12}{w[0]:>6.2f}{w[1]:>7.2f}{w[2]:>7.2f}{w[3]:>7.2f}"
              f"{m['macro_f1']:>10.4f}{m['level_accuracy']:>7.3f}{m['adjacent_accuracy']:>7.3f}"
              f"{(m['auc'] or 0):>7.3f}{m['binary_f1']:>8.3f}")
    gain = best_m["macro_f1"] - default_m["macro_f1"]
    print(f"    {'chênh lệch':<12}{'':>27}{gain:>+10.4f}"
          f"{best_m['level_accuracy'] - default_m['level_accuracy']:>+7.3f}")

    print("\n    Ma trận nhầm lẫn của bộ trọng số học được (hàng = giảng viên, cột = AI):")
    conf = best_m["confusion"]
    cols = list(conf)
    print("      " + " " * 12 + "".join(f"{c:>11}" for c in cols))
    for g in cols:
        print(f"      {g:<12}" + "".join(f"{conf[g][c]:>11}" for c in cols))

    io("OUTPUT",
       f"α_semantic = {best[0]:.2f}   β_lexical = {best[1]:.2f}   "
       f"γ_structure = {best[2]:.2f}   δ_domain = {best[3]:.2f}",
       f"macro-F1 {default_m['macro_f1']:.4f} → {best_m['macro_f1']:.4f} ({gain:+.4f})",
       f"tổng chi phí grid search: {size:,} ứng viên trong {elapsed:.3f}s")
    return best_m, default_m


# ── main ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Trace từng bước quá trình grid search học trọng số MDDM.")
    ap.add_argument("--step", type=float, default=0.05, help="bước nhảy của lưới (mặc định 0.05)")
    ap.add_argument("--top", type=int, default=10, help="số dòng bảng xếp hạng (mặc định 10)")
    ap.add_argument("--max-logs", type=int, default=25, help="số lần 'phá kỷ lục' được in ra")
    ap.add_argument("--save-weights", action="store_true",
                    help="ghi đè data/tuned_weights.json bằng trọng số vừa học")
    args = ap.parse_args()

    total = 8
    banner(" AI HỌC BỘ 4 TRỌNG SỐ (α, β, γ, δ) BẰNG GRID SEARCH ",
           " S_composite = α·S_sem + β·S_lex + γ·S_str + δ·S_dom ")

    queries, corpus, matrix, labelled = step1_load(total)
    samples = step2_matrix(total, queries, corpus, matrix, labelled)
    gold_dist = step3_labels(total, labelled)
    n, size = step4_space(total, args.step)
    best, improvements, all_points, elapsed = step5_scan(
        total, queries, matrix, n, size, args.max_logs)
    board = step6_leaderboard(total, all_points, args.top)
    marginals = step7_marginals(total, all_points, args.step)
    best_m, default_m = step8_final(total, queries, matrix, best, args.step, n, size, elapsed)

    payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "objective": "level macro-F1 (hoà: AUC → độ chính xác mức)",
        "step": args.step,
        "gridPoints": size,
        "searchSeconds": round(elapsed, 4),
        "dataset": {"corpusSize": len(corpus), "queries": len(queries),
                    "labelled": len(labelled), "goldLevelDist": gold_dist},
        "sampleFeatures": samples,
        "improvements": improvements,
        "leaderboard": board,
        "marginals": marginals,
        "best": {"weights": dict(zip(_DIMS, [round(x, 4) for x in best])),
                 "macroF1": best_m["macro_f1"], "accuracy": best_m["level_accuracy"],
                 "auc": best_m["auc"], "confusion": best_m["confusion"]},
        "paperDefault": {"weights": dict(zip(_DIMS, _DEFAULT_WEIGHTS)),
                         "macroF1": default_m["macro_f1"], "accuracy": default_m["level_accuracy"],
                         "auc": default_m["auc"]},
    }
    with open(_TRACE_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\n→ nhật ký đầy đủ: {_TRACE_FILE}")
    print("  (trang /demo đọc file này để vẽ lại quá trình học)")

    if args.save_weights:
        with open(_WEIGHTS_FILE, "w", encoding="utf-8") as handle:
            json.dump({
                "weights": dict(zip(_DIMS, [round(x, 4) for x in best])),
                "step": args.step,
                "macro_f1": best_m["macro_f1"],
                "level_accuracy": best_m["level_accuracy"],
                "source": "grid-search (level macro-F1) on data/FA26_topics_deduplicated.json",
            }, handle, indent=2, ensure_ascii=False)
        print(f"→ đã ghi {_WEIGHTS_FILE}  ·  bật bằng MDDM_USE_TUNED=true")


if __name__ == "__main__":
    main()
