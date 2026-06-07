#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delta_feature_analysis_Three.py

三分类差分分析：基于抗菌肽的 Exp_Log2_MIC 把样本分成"Low/Mid/High"三组，
Low ≤ 3, Mid > 3 且 ≤ 5, High > 5

对所有数值特征做差分分析（均值差、标准化效应量、相关性、Kruskal-Wallis检验），
并输出结果表与图片。

用法示例：
  python delta_feature_analysis_Three.py --input data.xlsx --outdir out

输出：
  out/
    delta_feature_summary.xlsx
    delta_feature_summary.csv
    figures/
      log2mic_hist.png
      top_delta_bar.png
      volcano.png
      corr_top20.png
      heatmap_top30.png
      group_comparison_boxplot/
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 尝试导入 scipy 用于 t-test 和 Kruskal-Wallis；若不可用则跳过 p-value
try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ----------------------------
# Constants - 三分类阈值
# ----------------------------
LOW_THRESHOLD = 3.0
MID_THRESHOLD = 5.0


# ----------------------------
# Utilities
# ----------------------------

COMMON_NONFEATURE_COLS = {
    "name", "id", "seq_id", "sequence_id", "sequence_name",
    "sequence", "sequences", "peptide", "peptide_sequence",
    "source", "label", "class", "group",
}

TARGET_CANDIDATES = [
    "Exp_Log2_MIC", "log2mic", "log2(mic)", "log2 mic",
    "exp_log2_mic", "mic_log2", "log2_mic_exp",
    "log2mic_exp", "log2_mic_value",
]


def _normalize_col(c: str) -> str:
    return str(c).strip().lower().replace("\u0394", "delta").replace("_", "").replace(" ", "")


def detect_target_column(df: pd.DataFrame, user_target: Optional[str] = None) -> str:
    if user_target:
        if user_target in df.columns:
            return user_target
        norm_map = { _normalize_col(c): c for c in df.columns }
        key = _normalize_col(user_target)
        if key in norm_map:
            return norm_map[key]
        raise ValueError(f"Target column '{user_target}' not found in input file.")

    norm_map = { _normalize_col(c): c for c in df.columns }
    for cand in TARGET_CANDIDATES:
        key = _normalize_col(cand)
        if key in norm_map:
            return norm_map[key]

    for c in df.columns:
        k = _normalize_col(c)
        if ("mic" in k) and ("log2" in k):
            return c

    raise ValueError(
        "Cannot auto-detect target column. Please pass --target (e.g., Exp_Log2_MIC)."
    )


def infer_numeric_features(df: pd.DataFrame, target_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c != target_col]

    drop = set()
    norm_non = {_normalize_col(x) for x in COMMON_NONFEATURE_COLS}
    for c in feats:
        if _normalize_col(c) in norm_non:
            drop.add(c)
    feats = [c for c in feats if c not in drop]

    good = []
    for c in feats:
        s = df[c]
        if s.notna().sum() < max(5, int(0.02 * len(df))):
            continue
        rng = float(s.max(skipna=True) - s.min(skipna=True))
        if rng == 0.0 or not np.isfinite(rng):
            continue
        good.append(c)
    return good


def split_groups_three(y: pd.Series, low_th: float = LOW_THRESHOLD, high_th: float = MID_THRESHOLD) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回三分类的 mask 数组：(low_mask, mid_mask, high_mask)

    Low:  y <= low_th
    Mid:  low_th < y <= high_th
    High: y > high_th
    """
    yv = y.values.astype(float)
    mask_valid = np.isfinite(yv)

    low_mask = mask_valid & (yv <= low_th)
    mid_mask = mask_valid & (yv > low_th) & (yv <= high_th)
    high_mask = mask_valid & (yv > high_th)

    return low_mask, mid_mask, high_mask


def cohen_d_pair(low: np.ndarray, high: np.ndarray, reference: str = "pooled") -> float:
    """
    计算两组的 Cohen's d
    reference: "pooled" - 使用 pooled SD; "low" - 以低组为参考; "high" - 以高组为参考
    """
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if len(low) < 2 or len(high) < 2:
        return np.nan

    if reference == "pooled":
        n1, n2 = len(low), len(high)
        v1, v2 = np.var(low, ddof=1), np.var(high, ddof=1)
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        if pooled <= 0:
            return np.nan
        return (np.mean(high) - np.mean(low)) / math.sqrt(pooled)
    elif reference == "low":
        sd = np.std(low, ddof=1)
        if sd <= 0:
            return np.nan
        return (np.mean(high) - np.mean(low)) / sd
    elif reference == "high":
        sd = np.std(high, ddof=1)
        if sd <= 0:
            return np.nan
        return (np.mean(high) - np.mean(low)) / sd
    return np.nan


def eta_squared(kruskal_result) -> float:
    """从 Kruskal-Wallis H 统计量计算 eta-squared"""
    if kruskal_result is None:
        return np.nan
    # eta_squared = (H - k + 1) / (n - k)
    # 其中 H 是 Kruskal-Wallis H, k 是组数, n 是总样本数
    return np.nan


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if SCIPY_OK:
        r, p = stats.pearsonr(x[mask], y[mask])
        return float(r), float(p)
    xm = x[mask] - np.mean(x[mask])
    ym = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    if denom == 0:
        return np.nan, np.nan
    r = float(np.sum(xm * ym) / denom)
    return r, np.nan


def safe_kruskal(low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> Tuple[float, float]:
    """
    Kruskal-Wallis H-test for three groups
    Returns: (H_statistic, p_value)
    """
    groups = []
    for g, name in [(low, "low"), (mid, "mid"), (high, "high")]:
        g_clean = g[np.isfinite(g)]
        if len(g_clean) >= 2:
            groups.append(g_clean)

    if len(groups) < 2:
        return np.nan, np.nan

    if SCIPY_OK:
        try:
            H, p = stats.kruskal(*groups)
            return float(H), float(p)
        except Exception:
            return np.nan, np.nan
    return np.nan, np.nan


def safe_ttest_pair(low: np.ndarray, high: np.ndarray) -> float:
    """Welch t-test for pair-wise comparison"""
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if len(low) < 2 or len(high) < 2:
        return np.nan
    if SCIPY_OK:
        try:
            _, p = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")
            return float(p)
        except Exception:
            return np.nan
    return np.nan


def safe_wilcoxon(low: np.ndarray, high: np.ndarray) -> Tuple[float, float]:
    """Mann-Whitney U test for pair-wise comparison"""
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if len(low) < 3 or len(high) < 3:
        return np.nan, np.nan
    if SCIPY_OK:
        try:
            stat, p = stats.mannwhitneyu(high, low, alternative='two-sided')
            return float(stat), float(p)
        except Exception:
            return np.nan, np.nan
    return np.nan, np.nan


# ----------------------------
# Plotting
# ----------------------------

def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_hist(y: pd.Series, out: Path, low_th: float = LOW_THRESHOLD, high_th: float = MID_THRESHOLD):
    """直方图，标注三分类阈值"""
    fig = plt.figure(figsize=(10, 6))
    vals = y.dropna().values.astype(float)

    # 绘制直方图
    plt.hist(vals, bins=40, alpha=0.7, edgecolor='black')

    # 添加阈值线
    ymin, ymax = plt.ylim()
    plt.axvline(x=low_th, color='green', linestyle='--', linewidth=2, label=f'Low/Mid: {low_th}')
    plt.axvline(x=high_th, color='red', linestyle='--', linewidth=2, label=f'Mid/High: {high_th}')

    # 统计各组数量
    n_low = (vals <= low_th).sum()
    n_mid = ((vals > low_th) & (vals <= high_th)).sum()
    n_high = (vals > high_th).sum()

    plt.title(f"Exp_Log2_MIC Distribution\nLow={n_low}, Mid={n_mid}, High={n_high}")
    plt.xlabel("Exp_Log2_MIC")
    plt.ylabel("Count")
    plt.legend()
    save_fig(fig, out)


def plot_group_distribution(y: pd.Series, out: Path, low_th: float = LOW_THRESHOLD, high_th: float = MID_THRESHOLD):
    """绘制三组分布的箱线图"""
    vals = y.dropna().values.astype(float)

    low_vals = vals[vals <= low_th]
    mid_vals = vals[(vals > low_th) & (vals <= high_th)]
    high_vals = vals[vals > high_th]

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [low_vals, mid_vals, high_vals]
    labels = [f'Low\n(n={len(low_vals)})\n≤{low_th}', f'Mid\n(n={len(mid_vals)})\n({low_th},{high_th}]', f'High\n(n={len(high_vals)})\n>{high_th}']

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Exp_Log2_MIC")
    ax.set_title("MIC Distribution by Three Groups")
    ax.grid(True, alpha=0.3, axis='y')
    save_fig(fig, out)


def plot_top_delta(summary: pd.DataFrame, out: Path, topn: int = 25):
    """绘制 Delta Mean (High - Low) 排名图"""
    df = summary.dropna(subset=["delta_high_low"]).copy()
    df = df.sort_values("abs_delta_high_low", ascending=False).head(topn)
    fig = plt.figure(figsize=(10, 8))
    plt.barh(df["feature"][::-1], df["delta_high_low"][::-1],
             color=['red' if x > 0 else 'blue' for x in df["delta_high_low"][::-1]])
    plt.title(f"Top {topn} |Delta Mean| (High - Low)")
    plt.xlabel("Delta Mean (High - Low)")
    plt.axvline(x=0, color='black', linewidth=0.5)
    save_fig(fig, out)


def plot_top_delta_mid(summary: pd.DataFrame, out: Path, topn: int = 25):
    """绘制 Mid-Low Delta Mean 排名图"""
    df = summary.dropna(subset=["delta_mid_low"]).copy()
    df = df.sort_values("abs_delta_mid_low", ascending=False).head(topn)
    fig = plt.figure(figsize=(10, 8))
    plt.barh(df["feature"][::-1], df["delta_mid_low"][::-1],
             color=['red' if x > 0 else 'blue' for x in df["delta_mid_low"][::-1]])
    plt.title(f"Top {topn} |Delta Mean| (Mid - Low)")
    plt.xlabel("Delta Mean (Mid - Low)")
    plt.axvline(x=0, color='black', linewidth=0.5)
    save_fig(fig, out)


def plot_volcano(summary: pd.DataFrame, out: Path, topn: int = 15):
    """火山图：Delta Mean vs -log10(p)"""
    df = summary.copy()
    df["neglog10_p"] = np.where(df["kruskal_p"].notna() & (df["kruskal_p"] > 0),
                               -np.log10(df["kruskal_p"]), np.nan)
    fig = plt.figure(figsize=(10, 8))
    x = df["delta_high_low"].values
    y = df["neglog10_p"].values
    plt.scatter(x, y, s=20, alpha=0.6)
    plt.title("Volcano: Delta Mean (High-Low) vs -log10(Kruskal-Wallis p)")
    plt.xlabel("Delta Mean (High - Low)")
    plt.ylabel("-log10(p)")

    # annotate topn
    score = df["abs_delta_high_low"].values.copy()
    if np.isfinite(y).any():
        yy = np.nan_to_num(y, nan=0.0)
        score = score * (1.0 + yy)
    idx = np.argsort(score)[-topn:]
    for i in idx:
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue
        plt.text(x[i], y[i], str(df.iloc[i]["feature"]), fontsize=7)

    save_fig(fig, out)


def plot_corr_top(summary: pd.DataFrame, out: Path, topn: int = 20):
    """绘制与 Exp_Log2_MIC Pearson 相关系数排名"""
    df = summary.dropna(subset=["pearson_r"]).copy()
    df = df.sort_values("abs_pearson_r", ascending=False).head(topn)
    fig = plt.figure(figsize=(10, 8))
    plt.barh(df["feature"][::-1], df["pearson_r"][::-1],
             color=['red' if x > 0 else 'blue' for x in df["pearson_r"][::-1]])
    plt.title(f"Top {topn} |Pearson r| with Exp_Log2_MIC")
    plt.xlabel("Pearson r")
    plt.axvline(x=0, color='black', linewidth=0.5)
    save_fig(fig, out)


def plot_group_comparison_boxplot(df: pd.DataFrame, feature: str, target_col: str,
                                   low_th: float, high_th: float, outdir: Path):
    """为单个特征绘制三组箱线图"""
    vals = pd.to_numeric(df[feature], errors="coerce")
    mask_low = vals <= low_th
    mask_mid = (vals > low_th) & (vals <= high_th)
    mask_high = vals > high_th

    low_data = vals[mask_low].dropna().values
    mid_data = vals[mask_mid].dropna().values
    high_data = vals[mask_high].dropna().values

    if len(low_data) < 2 and len(mid_data) < 2 and len(high_data) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [low_data, mid_data, high_data]
    labels = [f'Low\n(n={len(low_data)})', f'Mid\n(n={len(mid_data)})', f'High\n(n={len(high_data)})']

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    colors = ['#90EE90', '#FFD700', '#FF6B6B']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by MIC Group\n(Low<={low_th}, Mid>{low_th}<={high_th}, High>{high_th})")
    ax.grid(True, alpha=0.3, axis='y')

    safe_name = feature.replace(" ", "_").replace("/", "_")[:50]
    outpath = outdir / f"{safe_name}_boxplot.png"
    save_fig(fig, outpath)
    return outpath


def plot_heatmap(df: pd.DataFrame, features: List[str], target_col: str, out: Path,
                 low_th: float, high_th: float, max_rows: int = 4000):
    """热图：按 MIC 排序的样本"""
    sub = df[[target_col] + features].copy()
    sub = sub.dropna(subset=[target_col])
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=0)

    X = sub[features].astype(float)
    X = (X - X.mean()) / (X.std(ddof=0).replace(0, np.nan))
    y = sub[target_col].astype(float)

    order = np.argsort(y.values)
    X = X.iloc[order].values

    fig = plt.figure(figsize=(12, 8))
    plt.imshow(X, aspect="auto", interpolation="nearest", cmap='RdBu_r')
    plt.title(f"Heatmap (z-scored), rows sorted by Exp_Log2_MIC\n(Low<={low_th}, Mid>{low_th}<={high_th}, High>{high_th})")
    plt.xlabel("Features")
    plt.ylabel("Samples (sorted by MIC)")
    plt.xticks(ticks=np.arange(len(features)), labels=features, rotation=90, fontsize=7)
    save_fig(fig, out)


# ----------------------------
# Main Analysis
# ----------------------------

def run(input_path: str, outdir: str, target: Optional[str], topn: int,
        low_th: float = LOW_THRESHOLD, high_th: float = MID_THRESHOLD):
    """
    执行三分类差分分析

    Args:
        input_path: 输入数据文件路径
        outdir: 输出目录
        target: 目标列名（默认自动检测）
        topn: 可视化Top N特征
        low_th: Low组阈值（默认3.0）
        high_th: High组阈值（默认5.0）
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    boxplot_dir = figdir / "group_comparison_boxplot"
    boxplot_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    if input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    if df.empty:
        raise ValueError("Input file is empty.")

    # 检测目标列
    target_col = detect_target_column(df, target)

    # 转为数值型
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    y = df[target_col]

    # 推断数值特征
    feats = infer_numeric_features(df, target_col)
    if len(feats) == 0:
        raise ValueError("No numeric feature columns found (excluding target).")

    # 三分类分组
    low_m, mid_m, high_m = split_groups_three(y, low_th=low_th, high_th=high_th)
    n_low = int(low_m.sum())
    n_mid = int(mid_m.sum())
    n_high = int(high_m.sum())

    print(f"\n{'='*60}")
    print("Three-Class Group Split Statistics")
    print(f"{'='*60}")
    print(f"Thresholds: Low <= {low_th}, Mid ({low_th}, {high_th}], High > {high_th}")
    print(f"Low group:   {n_low:5d} samples ({n_low/len(df)*100:.1f}%)")
    print(f"Mid group:   {n_mid:5d} samples ({n_mid/len(df)*100:.1f}%)")
    print(f"High group:  {n_high:5d} samples ({n_high/len(df)*100:.1f}%)")
    print(f"{'='*60}\n")

    # 逐特征分析
    rows = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").values.astype(float)

        # 计算与 MIC 的相关性
        r, rp = safe_pearsonr(x, y.values.astype(float))

        # 提取三组数据
        lo = x[low_m]
        mi = x[mid_m]
        hi = x[high_m]

        # 各组统计量
        mean_low = float(np.nanmean(lo)) if np.isfinite(lo).any() else np.nan
        mean_mid = float(np.nanmean(mi)) if np.isfinite(mi).any() else np.nan
        mean_high = float(np.nanmean(hi)) if np.isfinite(hi).any() else np.nan

        std_low = float(np.nanstd(lo, ddof=1)) if np.isfinite(lo).any() else np.nan
        std_mid = float(np.nanstd(mi, ddof=1)) if np.isfinite(mi).any() else np.nan
        std_high = float(np.nanstd(hi, ddof=1)) if np.isfinite(hi).any() else np.nan

        # Delta 计算
        delta_high_low = mean_high - mean_low if (np.isfinite(mean_high) and np.isfinite(mean_low)) else np.nan
        delta_mid_low = mean_mid - mean_low if (np.isfinite(mean_mid) and np.isfinite(mean_low)) else np.nan
        delta_high_mid = mean_high - mean_mid if (np.isfinite(mean_high) and np.isfinite(mean_mid)) else np.nan

        # Cohen's d
        d_high_low = cohen_d_pair(lo, hi, reference="pooled")
        d_mid_low = cohen_d_pair(lo, mi, reference="pooled")
        d_high_mid = cohen_d_pair(mi, hi, reference="pooled")

        # Kruskal-Wallis 检验
        kw_H, kw_p = safe_kruskal(lo, mi, hi)

        # Pair-wise t-test (High vs Low)
        p_high_low = safe_ttest_pair(lo, hi)

        # Mann-Whitney U test (High vs Low)
        _, mw_p_high_low = safe_wilcoxon(lo, hi)

        row = {
            "feature": f,
            "n_total": int(np.isfinite(x).sum()),

            # 各组样本数
            "n_low": int(np.isfinite(lo).sum()),
            "n_mid": int(np.isfinite(mi).sum()),
            "n_high": int(np.isfinite(hi).sum()),

            # 各组均值
            "mean_low": mean_low,
            "mean_mid": mean_mid,
            "mean_high": mean_high,

            # 各组标准差
            "std_low": std_low,
            "std_mid": std_mid,
            "std_high": std_high,

            # Delta 值
            "delta_high_low": delta_high_low,
            "delta_mid_low": delta_mid_low,
            "delta_high_mid": delta_high_mid,

            "abs_delta_high_low": abs(delta_high_low) if np.isfinite(delta_high_low) else np.nan,
            "abs_delta_mid_low": abs(delta_mid_low) if np.isfinite(delta_mid_low) else np.nan,

            # Cohen's d
            "cohen_d_high_low": d_high_low,
            "cohen_d_mid_low": d_mid_low,
            "cohen_d_high_mid": d_high_mid,

            # 相关性
            "pearson_r": r,
            "pearson_p": rp,
            "abs_pearson_r": abs(r) if np.isfinite(r) else np.nan,

            # 统计检验
            "kruskal_H": kw_H,
            "kruskal_p": kw_p,
            "ttest_p_high_low": p_high_low,
            "mannwhitney_p_high_low": mw_p_high_low,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # 多重检验校正 (BH-FDR) - 使用 Kruskal-Wallis p-value
    if summary["kruskal_p"].notna().any():
        p = summary["kruskal_p"].values.astype(float)
        m = np.isfinite(p)
        pv = p[m]
        order = np.argsort(pv)
        ranked = pv[order]
        qvals = np.empty_like(ranked)
        n = len(ranked)
        prev = 1.0
        for i in range(n - 1, -1, -1):
            rank = i + 1
            val = ranked[i] * n / rank
            prev = min(prev, val)
            qvals[i] = prev
        qvals_full = np.full_like(p, np.nan)
        qvals_full[m] = qvals[np.argsort(order)]
        summary["fdr_bh_kruskal"] = qvals_full
    else:
        summary["fdr_bh_kruskal"] = np.nan

    # 按 delta_high_low 排序
    summary = summary.sort_values(["abs_delta_high_low", "abs_pearson_r"], ascending=[False, False])

    # 输出结果
    csv_path = out / "delta_feature_summary.csv"
    xlsx_path = out / "delta_feature_summary.xlsx"
    summary.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="summary")

        meta = pd.DataFrame([{
            "input": str(Path(input_path).resolve()),
            "target_col": target_col,
            "group_mode": "three_class",
            "low_threshold": low_th,
            "high_threshold": high_th,
            "n_rows": len(df),
            "n_low": n_low,
            "n_mid": n_mid,
            "n_high": n_high,
            "n_features": len(feats),
            "scipy_available": SCIPY_OK,
        }])
        meta.to_excel(w, index=False, sheet_name="meta")

    print(f"Results saved to: {xlsx_path}")
    print(f"Results saved to: {csv_path}")

    # 绘制图片
    print("\nGenerating plots...")
    plot_hist(y, figdir / "log2mic_hist.png", low_th=low_th, high_th=high_th)
    plot_group_distribution(y, figdir / "group_distribution.png", low_th=low_th, high_th=high_th)
    plot_top_delta(summary, figdir / "top_delta_bar.png", topn=topn)
    plot_top_delta_mid(summary, figdir / "top_delta_mid_bar.png", topn=topn)
    plot_volcano(summary, figdir / "volcano.png", topn=min(15, topn))
    plot_corr_top(summary, figdir / "corr_top20.png", topn=min(20, topn))

    # 绘制 Top 特征的箱线图
    top_feats = summary.dropna(subset=["abs_delta_high_low"]).head(min(30, len(summary)))["feature"].tolist()
    if len(top_feats) >= 5:
        plot_heatmap(df, top_feats, target_col, figdir / "heatmap_top30.png",
                    low_th=low_th, high_th=high_th)

        # 绘制 Top 特征的分组箱线图
        print(f"\nGenerating boxplots for top {min(20, len(top_feats))} features...")
        for i, feat in enumerate(top_feats[:20]):
            plot_group_comparison_boxplot(df, feat, target_col, low_th, high_th, boxplot_dir)

    print(f"\nPlots saved to: {figdir}/")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Target: {target_col}")
    print(f"Rows: {len(df)} | Features analyzed: {len(feats)}")
    print(f"Group split: Low<={low_th} (n={n_low}), Mid>{low_th}<={high_th} (n={n_mid}), High>{high_th} (n={n_high})")
    print(f"\nOutputs:")
    print(f"  {xlsx_path}")
    print(f"  {csv_path}")
    print(f"  {figdir}/")
    print("="*60)


def main():
    ap = argparse.ArgumentParser(
        description="三分类差分特征分析：Low/Mid/High (基于 Exp_Log2_MIC)\n"
                    f"  Low <= {LOW_THRESHOLD}\n"
                    f"  Mid > {LOW_THRESHOLD} 且 <= {MID_THRESHOLD}\n"
                    f"  High > {MID_THRESHOLD}"
    )
    ap.add_argument("--input", required=True, help="Input .xlsx or .csv file")
    ap.add_argument("--outdir", default="delta_out_three", help="Output directory")
    ap.add_argument("--target", default=None, help="Target column name (default: auto-detect Exp_Log2_MIC)")
    ap.add_argument("--low_th", type=float, default=LOW_THRESHOLD,
                   help=f"Threshold for Low group (default: {LOW_THRESHOLD})")
    ap.add_argument("--high_th", type=float, default=MID_THRESHOLD,
                   help=f"Threshold for High group (default: {MID_THRESHOLD})")
    ap.add_argument("--topn", type=int, default=25, help="Top N features to visualize")
    args = ap.parse_args()

    run(
        input_path=args.input,
        outdir=args.outdir,
        target=args.target,
        topn=args.topn,
        low_th=args.low_th,
        high_th=args.high_th,
    )


if __name__ == "__main__":
    main()
