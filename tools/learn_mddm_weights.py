#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learn_mddm_weights.py
=====================
Học trọng số (alpha, beta, gamma, delta) cho mô hình hợp nhất MDDM của DASSF
trên một validation split, thay vì gán cứng cảm tính.

S_composite = alpha*S_sem + beta*S_lex + gamma*S_str + delta*S_dom
ràng buộc: alpha + beta + gamma + delta = 1, mọi trọng số >= 0  (simplex)

Script cung cấp 3 cách học trọng số để đối chứng lẫn nhau:
  (1) Logistic Regression  -> lấy hệ số, chiếu lên simplex (clip âm về 0, chuẩn hoá tổng = 1)
  (2) Constrained optimize -> tối ưu trực tiếp trên simplex để cực đại F1 trên validation
  (3) Grid search          -> quét lưới trên simplex (kiểm tra chéo, không phụ thuộc mô hình)

Ngưỡng quyết định tau được chọn để cực đại F1 trên validation (đúng như methodology
"the decision threshold that maximizes F1 is selected" trong bài), rồi báo cáo F1 trên test.

CÁCH DÙNG
---------
  # Dữ liệu thật của bạn (CSV hoặc XLSX) với 4 cột điểm + 1 cột nhãn 0/1:
  python learn_mddm_weights.py --input scored_pairs.xlsx \
      --col-sem s_sem --col-lex s_lex --col-str s_str --col-dom s_dom --col-label label

  # Chạy thử với dữ liệu giả lập để xem định dạng đầu ra:
  python learn_mddm_weights.py --demo

ĐỊNH DẠNG FILE ĐẦU VÀO
----------------------
Một dòng = một cặp đề tài, gồm 4 điểm tương đồng theo từng chiều (đã chuẩn hoá về [0,1])
và nhãn nhị phân (1 = trùng lặp cấu trúc, 0 = không), ví dụ:

  pair_id, s_sem, s_lex, s_str, s_dom, label
  P0001,   0.41,  0.33,  0.98,  0.19,  1
  P0002,   0.12,  0.08,  0.15,  0.05,  0
  ...
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import minimize

DIMS = ["sem", "lex", "str", "dom"]
GREEK = {"sem": "alpha", "lex": "beta", "str": "gamma", "dom": "delta"}


# --------------------------------------------------------------------------- #
# Tiện ích                                                                     #
# --------------------------------------------------------------------------- #
def best_threshold(scores, y):
    """Tìm ngưỡng tau cực đại F1; trả về (tau, f1, precision, recall)."""
    cand = np.unique(scores)
    # thêm điểm giữa để ngưỡng nằm giữa hai giá trị liền kề
    mids = (cand[:-1] + cand[1:]) / 2.0 if len(cand) > 1 else cand
    cand = np.concatenate([[-1e-9], mids, [1.0 + 1e-9]])
    best = (0.5, -1.0, 0.0, 0.0)
    for t in cand:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best[1]:
            best = (float(t),
                    float(f1),
                    float(precision_score(y, pred, zero_division=0)),
                    float(recall_score(y, pred, zero_division=0)))
    return best


def project_to_simplex_nonneg(coef):
    """Chiếu hệ số lên simplex: cắt phần âm về 0 rồi chuẩn hoá tổng = 1."""
    w = np.clip(np.asarray(coef, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:                      # phòng trường hợp suy biến
        return np.ones_like(w) / len(w)
    return w / s


def composite(X, w):
    return X @ np.asarray(w, dtype=float)


def evaluate(w, X_val, y_val, X_test, y_test):
    """Chọn tau trên validation rồi báo cáo trên test."""
    tau, f1_val, _, _ = best_threshold(composite(X_val, w), y_val)
    pred_test = (composite(X_test, w) >= tau).astype(int)
    return {
        "tau": tau,
        "f1_val": f1_val,
        "precision_test": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall_test": float(recall_score(y_test, pred_test, zero_division=0)),
        "f1_test": float(f1_score(y_test, pred_test, zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# Ba cách học trọng số                                                         #
# --------------------------------------------------------------------------- #
def weights_logreg(X_tr, y_tr):
    """(1) Logistic Regression -> chiếu hệ số lên simplex."""
    clf = LogisticRegression(C=1.0, fit_intercept=True,
                             max_iter=5000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return project_to_simplex_nonneg(clf.coef_.ravel())


def weights_constrained(X_tr, y_tr):
    """(2) Tối ưu log-loss trực tiếp trên simplex (w>=0, sum=1)."""
    eps = 1e-9

    def neg_loglik(w):
        z = X_tr @ w
        p = 1.0 / (1.0 + np.exp(-(8.0 * (z - 0.5))))  # logistic quanh 0.5
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p))

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * 4
    w0 = np.full(4, 0.25)
    res = minimize(neg_loglik, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return project_to_simplex_nonneg(res.x)


def weights_grid(X_val, y_val, step=0.05):
    """(3) Grid search trên simplex; chọn bộ trọng số cực đại F1 (val)."""
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    best_w, best_f1 = np.full(4, 0.25), -1.0
    for a in grid:
        for b in grid:
            if a + b > 1.0 + 1e-9:
                continue
            for c in grid:
                d = 1.0 - a - b - c
                if d < -1e-9 or d > 1.0 + 1e-9:
                    continue
                w = np.array([a, b, c, max(d, 0.0)])
                _, f1, _, _ = best_threshold(composite(X_val, w), y_val)
                if f1 > best_f1:
                    best_f1, best_w = f1, w
    return best_w


# --------------------------------------------------------------------------- #
# Nạp dữ liệu                                                                  #
# --------------------------------------------------------------------------- #
def load_scores(path, cols):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        sys.exit(f"[LỖI] Thiếu cột trong file: {missing}\nCác cột hiện có: {list(df.columns)}")
    X = df[[cols["sem"], cols["lex"], cols["str"], cols["dom"]]].to_numpy(float)
    y = df[cols["label"]].to_numpy(int)
    return X, y


def make_demo(n=318, seed=42):
    """Dữ liệu giả lập CHỈ để minh hoạ định dạng đầu ra — KHÔNG phải số liệu thật."""
    rng = np.random.default_rng(seed)
    half = n // 2
    # Positive (trùng cấu trúc): str cao, dom thấp (giống benchmark semi-synthetic)
    pos = np.column_stack([
        rng.normal(0.45, 0.12, half),   # sem
        rng.normal(0.40, 0.12, half),   # lex
        rng.normal(0.95, 0.05, half),   # str  (cao)
        rng.normal(0.20, 0.08, half),   # dom  (thấp)
    ])
    # Negative
    neg = np.column_stack([
        rng.normal(0.20, 0.12, n - half),
        rng.normal(0.18, 0.12, n - half),
        rng.normal(0.25, 0.15, n - half),
        rng.normal(0.15, 0.10, n - half),
    ])
    X = np.clip(np.vstack([pos, neg]), 0, 1)
    y = np.array([1] * half + [0] * (n - half))
    idx = rng.permutation(n)
    return X[idx], y[idx]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Học trọng số MDDM trên validation split.")
    ap.add_argument("--input", help="File CSV/XLSX chứa điểm 4 chiều + nhãn.")
    ap.add_argument("--demo", action="store_true", help="Chạy với dữ liệu giả lập.")
    ap.add_argument("--col-sem", default="s_sem")
    ap.add_argument("--col-lex", default="s_lex")
    ap.add_argument("--col-str", default="s_str")
    ap.add_argument("--col-dom", default="s_dom")
    ap.add_argument("--col-label", default="label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-step", type=float, default=0.05)
    args = ap.parse_args()

    if args.demo or not args.input:
        if not args.demo:
            print("[Lưu ý] Không có --input nên chạy chế độ --demo (dữ liệu giả lập).\n")
        X, y = make_demo(seed=args.seed)
        demo = True
    else:
        cols = {"sem": args.col_sem, "lex": args.col_lex, "str": args.col_str,
                "dom": args.col_dom, "label": args.col_label}
        X, y = load_scores(args.input, cols)
        demo = False

    # Split 60/20/20 (stratified): train học trọng số, val chọn tau + grid, test báo cáo
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed)

    print(f"Số cặp: train={len(y_tr)}  val={len(y_val)}  test={len(y_test)}  "
          f"(tỉ lệ positive = {y.mean():.2f})\n")

    methods = {
        "Logistic Regression -> simplex": weights_logreg(X_tr, y_tr),
        "Constrained optimize (SLSQP)  ": weights_constrained(X_tr, y_tr),
        "Grid search on validation     ": weights_grid(X_val, y_val, args.grid_step),
    }

    print(f"{'Phương pháp':32s} | {'alpha':>6s} {'beta':>6s} {'gamma':>6s} {'delta':>6s} "
          f"| {'tau':>5s} | {'F1(test)':>8s}")
    print("-" * 86)
    for name, w in methods.items():
        ev = evaluate(w, X_val, y_val, X_test, y_test)
        print(f"{name:32s} | {w[0]:6.2f} {w[1]:6.2f} {w[2]:6.2f} {w[3]:6.2f} "
              f"| {ev['tau']:5.2f} | {ev['f1_test']:8.3f}")

    # Bộ trọng số khuyến nghị: lấy từ Logistic Regression (đúng như mô tả trong bài)
    w_lr = methods["Logistic Regression -> simplex"]
    ev_lr = evaluate(w_lr, X_val, y_val, X_test, y_test)
    print("\n>>> Trọng số học được (Logistic Regression), làm tròn 2 chữ số:")
    for i, dim in enumerate(DIMS):
        print(f"      {GREEK[dim]:6s} ({dim}) = {round(float(w_lr[i]), 2)}")
    print(f"      tau = {round(ev_lr['tau'], 2)}")
    print(f"      Test: P={ev_lr['precision_test']:.3f}  "
          f"R={ev_lr['recall_test']:.3f}  F1={ev_lr['f1_test']:.3f}")

    if demo:
        print("\n[CHÚ Ý] Đây là dữ liệu GIẢ LẬP để minh hoạ định dạng. "
              "Hãy chạy lại với --input là điểm số THẬT của bạn rồi mới điền số vào bài.")


if __name__ == "__main__":
    main()
