#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_lowmic_filter.py

合并Step3和Step4的功能，通过HCS_Dynamic_V3.py调用程序获得HMom_HoF特征，
然后应用以下过滤规则：

    0.330 ≤ HMom_HoF ≤ 0.470
    0.038 ≤ HCS3_HoF_Mean_SD ≤ 0.233

功能说明：
1) 支持读取 .txt / .tsv / .csv
2) 如果没有输入文件，正常跳过
3) 调用 HCS_Dynamic_V3.py 的 calculate_hcs_features 函数获取特征
4) 规则过滤后保存通过的多肽

过滤规则（规则和动态显示保存不变）：
    0.330 ≤ HMom_HoF ≤ 0.470
    0.038 ≤ HCS3_HoF_Mean_SD ≤ 0.233
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import Iterator, Tuple

import pandas as pd

# 导入HCS_Dynamic_V3的计算函数
from HCS_Dynamic import calculate_hcs_features


AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid_sequence(seq: str) -> bool:
    """检查序列是否有效"""
    seq = str(seq).strip().upper()
    return bool(seq) and all(c in AA_SET for c in seq)


def parse_input_file(path: str) -> Iterator[Tuple[str, str]]:
    """
    支持读取：
    1) name\tseq
    2) Name / Sequences 表格
    3) 普通 txt/csv/tsv 表格
    4) 纯序列文件
    """
    ext = os.path.splitext(path)[1].lower()

    # ---------- CSV / TSV 表格优先用 pandas ----------
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        try:
            df = pd.read_csv(path, sep=sep)
            cols = list(df.columns)

            # 标准列名
            if "Name" in cols and "Sequences" in cols:
                for i, r in df.iterrows():
                    name = str(r.get("Name", f"Seq_{i + 1}")).strip()
                    seq = str(r.get("Sequences", "")).strip().upper()
                    if is_valid_sequence(seq):
                        yield name if name else f"Seq_{i + 1}", seq
                return

            # 至少两列，默认第1列为name，第2列为seq
            if len(cols) >= 2:
                for i, r in df.iterrows():
                    name = str(r.iloc[0]).strip()
                    seq = str(r.iloc[1]).strip().upper()
                    if is_valid_sequence(seq):
                        yield name if name else f"Seq_{i + 1}", seq
                return

            # 只有一列，尝试作为纯序列
            if len(cols) == 1:
                for i, r in df.iterrows():
                    seq = str(r.iloc[0]).strip().upper()
                    if is_valid_sequence(seq):
                        yield f"Seq_{i + 1}", seq
                return

        except Exception:
            # 若 pandas 读取失败，回退到普通文本读取
            pass

    # ---------- TXT / 回退解析 ----------
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 跳过常见表头
            lower = line.lower()
            if idx == 1 and ("sequence" in lower or "sequences" in lower):
                continue

            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = [line]

            if len(parts) >= 2:
                name = parts[0].strip()
                seq = parts[1].strip().upper()
            else:
                name = f"Seq_{idx}"
                seq = parts[0].strip().upper()

            if is_valid_sequence(seq):
                yield name if name else f"Seq_{idx}", seq


def step3_pass(name: str, seq: str) -> bool:
    """
    Step3/Step4 合并后的过滤规则：

    通过 HCS_Dynamic_V3.py 的 calculate_hcs_features 函数获取：
    - HMom_HoF: 疏水面平均疏水距
    - HCS3_HoF_Mean_SD: HCS3疏水面疏水距标准差

    过滤规则：
        0.330 ≤ HMom_HoF ≤ 0.470
        0.038 ≤ HCS3_HoF_Mean_SD ≤ 0.233

    """
    try:
        # 调用 HCS_Dynamic_V3.py 获取特征
        rec = calculate_hcs_features(seq, name)

        # 获取 HMom_HoF
        hmom_hf = float(rec.get("HMom_HoF", 0.0))

        # 应用过滤规则
        # 0.330 ≤ HMom_HoF ≤ 0.470
        if not (0.330 <= hmom_hf <= 0.470):
            return False

        # 获取 HCS3_HoF_Mean_SD
        hcs3_hf_mean_sd = float(rec.get("HCS3_HoF_Mean_SD", 0.0))

        # 应用过滤规则
        # 0.038 ≤ HCS3_HoF_Mean_SD ≤ 0.233
        if not (0.038 <= hcs3_hf_mean_sd <= 0.233):
            return False

        return True

    except Exception as e:
        # 如果计算失败，不通过
        return False


def process_file_to_queue(infile: str, q, report_every: int):
    """处理单个文件，将通过的多肽放入队列"""
    processed = 0
    passed = 0

    for name, seq in parse_input_file(infile):
        processed += 1

        if step3_pass(name, seq):
            passed += 1
            q.put(("hit", f"{name}\t{seq}\n"))

        if processed % report_every == 0:
            q.put(("file_stat", os.path.basename(infile), processed, passed))

    q.put(("file_stat", os.path.basename(infile), processed, passed))


def worker_file(job_q, out_q, report_every: int):
    """工作进程：从作业队列获取文件并处理"""
    while True:
        infile = job_q.get()
        if infile is None:
            break

        process_file_to_queue(infile, out_q, report_every)

    out_q.put(("done", 1))


def writer_step(outdir: str, chunk_records: int, q, n_workers: int, total_files: int):
    """
    写入器进程：从输出队列获取结果并写入文件

    规则和动态显示保存不变
    """
    os.makedirs(outdir, exist_ok=True)

    done = 0
    total_saved = 0
    file_idx = 0
    records_in_file = 0
    f = None
    file_stats = {}
    t0 = time.time()

    def open_new_file(idx: int):
        path = os.path.join(outdir, f"part_{idx:06d}.txt")
        return open(
            path,
            "w",
            encoding="utf-8",
            newline="\n",
            buffering=8 * 1024 * 1024,
        )

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
                f"\r[Step3_LowMIC] files_done={len(file_stats)}/{total_files}"
                f" | processed={total_processed:,}"
                f" | passed={total_passed:,}"
                f" | saved={total_saved:,}"
                f" | files={file_idx + (1 if f is not None else 0)}"
            )
            sys.stderr.flush()

        elif tag == "done":
            done += 1
            if done >= n_workers:
                break

    if f is not None:
        f.close()

    dt = time.time() - t0
    sys.stderr.write(
        f"\n✅ Step3_LowMIC done | saved={total_saved:,} | files={file_idx} | time={dt / 60:.2f} min\n"
    )
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(
        description="Step3/Step4 合并过滤：使用 HCS_Dynamic_V3.py 获取 HMom_HoF 和 HCS3_HoF_Mean_SD 并应用规则过滤"
    )
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chunk", type=int, default=1000000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--report_every", type=int, default=2000)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.isdir(args.indir):
        print(f"[Step3_LowMIC] Input directory does not exist: {args.indir}")
        print("[Step3_LowMIC] Skip Step3.")
        return

    files = sorted(
        os.path.join(args.indir, f)
        for f in os.listdir(args.indir)
        if f.lower().endswith((".txt", ".tsv", ".csv"))
    )

    if not files:
        print(f"[Step3_LowMIC] No input files found in: {args.indir}")
        print("[Step3_LowMIC] Skip Step3 because Step2 produced no candidate files.")
        return

    workers = max(1, min(args.workers, len(files)))

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    out_q = ctx.Queue(10000)

    wr = ctx.Process(
        target=writer_step,
        args=(args.outdir, args.chunk, out_q, workers, len(files)),
    )
    wr.start()

    ws = []
    for _ in range(workers):
        p = ctx.Process(
            target=worker_file,
            args=(job_q, out_q, args.report_every),
        )
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