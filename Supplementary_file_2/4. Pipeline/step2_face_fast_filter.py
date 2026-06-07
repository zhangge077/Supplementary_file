#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import multiprocessing as mp
import os
import sys
import time
import math

from PeptidePipeline_Core_V3 import find_helical_faces_multiround

# =========================================
# 基础参数
# =========================================
BASE_TARGET = [10, 3, 14, 7, 0, 18]

# =========================================
# 工具函数
# =========================================
def calc_face_center(indices):
    if not indices:
        return None
    angles = [math.radians(i * 100) for i in indices]
    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)
    return math.degrees(math.atan2(y, x)) % 360

def angle_diff(a, b):
    d = abs(a - b)
    return min(d, 360 - d)

# =========================================
# Step2-Pro核心函数
# =========================================
def step2_pass(seq: str) -> bool:
    try:
        hyd_seq, phil_seq, hyd_wheel, phil_wheel, hf_lin, pf_lin = \
            find_helical_faces_multiround(seq)
    except Exception:
        return False

    # 1️⃣ 必须有疏水面
    if not hf_lin:
        return False
    
    # 2️⃣ 检查疏水面是否严格等于 BASE_TARGET
    if hf_lin != BASE_TARGET:
        return False

    # 3️⃣ 疏水面中不能包含 A
    if any(seq[i] == "A" for i in hf_lin):
        return False

    # 4️⃣ 必须有亲水面
    if not pf_lin:
        return False
    pf = sorted({i for i in pf_lin if i < 19})

    # 5️⃣ 亲水面至少3个残基
    if len(pf) < 3:
        return False

    # 6️⃣ 亲水面必须含K或R
    if not any(seq[i] in {"K", "R"} for i in pf):
        return False

    # 7️⃣ 面对立（≈180°）
    h_center = calc_face_center(hf_lin)
    p_center = calc_face_center(pf)
    if h_center is None or p_center is None:
        return False
    diff = angle_diff(h_center, p_center)
    if diff < 120:
        return False

    return True

# =========================================
# IO及多进程逻辑
# =========================================
def parse_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                name, seq = line.split("\t", 1)
                yield name, seq
            except ValueError:
                continue

def process_file_to_queue(infile, q, report_every):
    processed = 0
    passed = 0
    for name, seq in parse_txt(infile):
        processed += 1
        if step2_pass(seq):
            passed += 1
            q.put(("hit", f"{name}\t{seq}\n"))
        if processed % report_every == 0:
            q.put(("file_stat", os.path.basename(infile), processed, passed))
    q.put(("file_stat", os.path.basename(infile), processed, passed))

def worker_file(job_q, out_q, report_every):
    while True:
        infile = job_q.get()
        if infile is None:
            break
        process_file_to_queue(infile, out_q, report_every)
    out_q.put(("done", 1))

def writer_step(outdir, chunk_records, q, n_workers, total_files):
    os.makedirs(outdir, exist_ok=True)
    done = 0
    total_saved = 0
    file_idx = 0
    records_in_file = 0
    f = None
    file_stats = {}
    t0 = time.time()

    def open_new_file(idx):
        path = os.path.join(outdir, f"part_{idx:06d}.txt")
        return open(path, "w", encoding="utf-8", buffering=8 * 1024 * 1024)

    while True:
        msg = q.get()
        tag = msg[0]

        if tag == "hit":
            line = msg[1]
            if f is None:
                file_idx += 1
                f = open_new_file(file_idx)
                records_in_file = 0
            f.write(line)
            total_saved += 1
            records_in_file += 1
            if records_in_file >= chunk_records:
                f.close()
                f = None
                records_in_file = 0

        elif tag == "file_stat":
            _, fname, processed, passed = msg
            file_stats[fname] = (processed, passed)
            total_processed = sum(x[0] for x in file_stats.values())
            total_passed = sum(x[1] for x in file_stats.values())
            sys.stderr.write(
                f"\r[Step2] files_done={len(file_stats)}/{total_files}"
                f" | processed={total_processed:,}"
                f" | passed={total_passed:,}"
                f" | saved={total_saved:,}"
            )
            sys.stderr.flush()

        elif tag == "done":
            done += 1
            if done >= n_workers:
                break

    if f:
        f.close()
    dt = time.time() - t0
    print(f"\n✅ Step2 done | saved={total_saved:,} | time={dt/60:.2f} min")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=os.cpu_count()-1)
    ap.add_argument("--report_every", type=int, default=5000)
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.indir, f)
        for f in os.listdir(args.indir)
        if f.endswith(".txt")
    )

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    out_q = ctx.Queue(10000)

    wr = ctx.Process(target=writer_step,
                     args=(args.outdir, args.chunk, out_q, args.workers, len(files)))
    wr.start()

    ws = []
    for _ in range(args.workers):
        p = ctx.Process(target=worker_file,
                        args=(job_q, out_q, args.report_every))
        p.start()
        ws.append(p)

    for fp in files:
        job_q.put(fp)
    for _ in ws:
        job_q.put(None)

    for w in ws:
        w.join()
    wr.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()