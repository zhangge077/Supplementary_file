#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import multiprocessing as mp
import os
import sys
import time

from PeptidePipeline_Core_V3 import (
    calculate_mean_hydrophobicity,
    calculate_mean_amphipathic_moment,
    calculate_net_charge,
)

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


# =========================
# 氨基酸频率阈值定义（统一使用宽松阈值）
# =========================

# 基础阈值（通用）
FREQ_A_MAX = 0.110      # 丙氨酸
FREQ_Q_MAX = 0.056      # 谷氨酰胺
FREQ_W_MAX = 0.160     # 色氨酸
FREQ_F_MAX = 0.150     # 苯丙氨酸
FREQ_L_MAX = 0.4445   # 亮氨酸
FREQ_I_MAX = 0.3889     # 异亮氨酸


def get_threshold_values():
    """
    
    返回：
        dict: 氨基酸 -> 频率阈值
    """
    return {
        'A': FREQ_A_MAX,
        'Q': FREQ_Q_MAX,
        'W': FREQ_W_MAX,
        'F': FREQ_F_MAX,
        'L': FREQ_L_MAX,
        'I': FREQ_I_MAX,
    }


# =========================
# 高速 idx → combo（核心加速）
# =========================
def idx_to_combo(idx, aa, width):
    base = len(aa)
    out = [None] * width
    for i in range(width - 1, -1, -1):
        idx, r = divmod(idx, base)
        out[i] = aa[r]
    return out


# =========================
# 解析 model（支持多 *）
# =========================
def parse_model(model: str):
    model = model.strip().upper()
    if len(model) != 19:
        raise ValueError(f"--model length must be 19, got {len(model)}")

    fixed = list(model)
    star_pos = []

    for i, c in enumerate(model):
        if c == "*":
            star_pos.append(i)
        elif c not in AA20:
            raise ValueError(f"invalid model char at position {i+1}: {c}")

    if not star_pos:
        raise ValueError("model must contain at least one '*'")

    return fixed, star_pos


# =========================
# aa_lib
# =========================
def validate_aa_lib(aa_lib: str):
    aa = []
    seen = set()
    for c in aa_lib.strip().upper():
        if c not in AA20:
            raise ValueError(f"invalid AA in aa_lib: {c}")
        if c not in seen:
            aa.append(c)
            seen.add(c)
    if not aa:
        raise ValueError("aa_lib is empty")
    return aa


# =========================
# Step1过滤（仅Core_V3基础函数）
# =========================
def step1_filter(seq: str) -> bool:
    """
    Step1 过滤函数

    使用固定的宽松阈值进行筛选

    筛选规则：
    1. 序列长度必须为 19
    2. 不能包含 D、E、P 氨基酸
    3. 氨基酸频率限制（使用宽松阈值）
    4. 净电荷 1.0 <= z <= 10.0
    5. 平均疏水性 0 <= hyd <= 0.52
    6. 平均两亲性 hm >= 0.49

    """
    # 获取固定阈值
    thresholds = get_threshold_values()
    seq_len = len(seq)

    if len(seq) != 19:
        return False
    if any(a in seq for a in "DEP"):
        return False

    # 遍历所有定义的阈值进行频率检查
    for aa, max_freq in thresholds.items():
        if seq.count(aa) > int(max_freq*seq_len):
            return False

    try:
        hyd = float(calculate_mean_hydrophobicity(seq))
        hm = float(calculate_mean_amphipathic_moment(seq))
        z = float(calculate_net_charge(seq))
    except Exception:
        return False

    if not (1.0 <= z <= 10.0):
        return False
    if not (0 <= hyd <= 0.52):
        return False
    if not (0.49 <= hm):
        return False

    return True


# =========================
# Worker（Old速度 + 新逻辑）
# =========================
def worker_enum_step1(wid, start, end, aa, star_n, fixed, star_pos, q, report_every):

    processed = 0
    passed = 0
    t0 = time.time()

    for i in range(start, end):

        combo = idx_to_combo(i, aa, star_n)

        seq_list = fixed.copy()
        for j, pos in enumerate(star_pos):
            seq_list[pos] = combo[j]

        seq = "".join(seq_list)

        processed += 1

        if step1_filter(seq):
            passed += 1
            q.put(("hit", f"Enum_{i+1}\t{seq}\n"))

        if processed % report_every == 0:
            speed = processed / max(time.time() - t0, 1e-9)
            q.put(("stat", wid, processed, passed, speed))

    q.put(("stat", wid, processed, passed, processed / max(time.time() - t0, 1e-9)))
    q.put(("done", wid))


# =========================
# writer（固定 workers-1 文件）
# =========================
def writer_step(outdir, chunk_records, q, n_workers, total_space, workers):

    os.makedirs(outdir, exist_ok=True)

    shard_n = max(1, workers - 1)

    # 预先创建固定文件
    files = [
        open(os.path.join(outdir, f"part_{i+1:06d}.txt"),
             "w", encoding="utf-8", newline="\n", buffering=8 * 1024 * 1024)
        for i in range(shard_n)
    ]

    total_saved = 0
    done = 0
    worker_stats = {}
    t0 = time.time()

    while True:
        msg = q.get()
        tag = msg[0]

        if tag == "hit":
            line = msg[1]

            # 平均分配（核心）
            idx = total_saved % shard_n
            files[idx].write(line)

            total_saved += 1

        elif tag == "stat":
            _, wid, processed, passed, speed = msg
            worker_stats[wid] = (processed, passed, speed)

            total_processed = sum(x[0] for x in worker_stats.values())
            total_passed = sum(x[1] for x in worker_stats.values())
            total_speed = sum(x[2] for x in worker_stats.values())

            pct = total_processed / total_space * 100 if total_space else 0.0

            sys.stderr.write(
                f"\r[Step1] processed={total_processed:,}/{total_space:,} ({pct:.4f}%)"
                f" | passed={total_passed:,}"
                f" | saved={total_saved:,}"
                f" | speed={total_speed:,.0f}/s"
                f" | files={shard_n}"
            )
            sys.stderr.flush()

        elif tag == "done":
            done += 1
            if done >= n_workers:
                break

    for f in files:
        f.close()

    dt = time.time() - t0
    sys.stderr.write(
        f"\n✅ Step1 done | saved={total_saved:,} | files={shard_n} | time={dt/60:.2f} min\n"
    )
    sys.stderr.flush()


# =========================
# main
# =========================
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--aa_lib", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk", type=int, default=10000000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--report_every", type=int, default=10000)
    args = ap.parse_args()

    aa = validate_aa_lib(args.aa_lib)

    fixed, star_pos = parse_model(args.model)
    star_n = len(star_pos)

    total = len(aa) ** star_n
    chunk = total // args.workers

    print(f"[*] star positions: {star_pos}")
    print(f"[*] combinations: {len(aa)}^{star_n} = {total}")

    # 显示当前使用的阈值
    thresholds = get_threshold_values()
    print(f"[*] thresholds: {thresholds}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue(10000)

    wr = ctx.Process(
        target=writer_step,
        args=(args.outdir, args.chunk, q, args.workers, total, args.workers)
    )
    wr.start()

    ws = []
    for i in range(args.workers):
        s = i * chunk
        e = (i + 1) * chunk if i < args.workers - 1 else total

        p = ctx.Process(
            target=worker_enum_step1,
            args=(i, s, e, aa, star_n, fixed, star_pos, q, args.report_every)
        )
        p.start()
        ws.append(p)

    for w in ws:
        w.join()
    wr.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()