

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python run_amp_trillion_pipeline_V2.py --model M****************GG --aa_lib KAWL --out_root KAWL_Model
# 清理模式: python run_amp_trillion_pipeline_V2.py --model M****************GG --aa_lib KAWL --out_root KAWL_Model --clean

import os
import json
import itertools
import subprocess
import shutil


MAX_ENUM = 1_500_000_000


# =========================
# 基础工具
# =========================
def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def recreate_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def load_progress(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {"done": []}


def save_progress(path, data):
    json.dump(data, open(path, "w"), indent=2)


# =========================
# 模型拆分
# =========================
def auto_split_star(model, aa_lib):

    star_idx = [i for i, c in enumerate(model) if c == "*"]
    total_star = len(star_idx)
    base = len(aa_lib)

    for remain in range(total_star, 0, -1):
        if base ** remain <= MAX_ENUM:
            fix_n = total_star - remain
            return star_idx[:fix_n], remain

    return star_idx[:3], total_star - 3


def generate_templates(model, aa_lib, fix_pos):

    combos = itertools.product(aa_lib, repeat=len(fix_pos))
    templates = []

    for combo in combos:
        m = list(model)
        for i, pos in enumerate(fix_pos):
            m[pos] = combo[i]
        templates.append("".join(m))

    return templates


# =========================
# 单模板 Candidates 输出
# =========================
def append_candidates(src_dir, out_file):

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))

    with open(out_file, "w") as out:

        for f in files:
            with open(os.path.join(src_dir, f)) as r:
                for line in r:
                    if not line.strip():
                        continue
                    out.write(line)


# =========================
# 🔥 全部 Candidates 合并 + 拆分 + 重新编号
# =========================
def merge_all_candidates(cand_dir, out_dir, chunk=1_000_000):

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(cand_dir) if f.endswith("_Candidates.txt"))

    file_idx = 0
    count = 0
    out = None

    def new_file():
        nonlocal file_idx
        file_idx += 1
        return open(os.path.join(out_dir, f"ALL_{file_idx:06d}.txt"), "w")

    for f in files:
        with open(os.path.join(cand_dir, f)) as r:
            for line in r:
                if not line.strip():
                    continue

                if out is None:
                    out = new_file()

                out.write(line)
                count += 1

                if count >= chunk:
                    out.close()
                    out = None
                    count = 0

    if out:
        out.close()


# =========================
# 🔥 重新编号 + 生成 Name Sequences 和 FASTA
# =========================
def renumber_and_export_sequences(all_dir, out_root):
    """
    合并所有 ALL_*.txt 文件，重新编号为 Enum_* 格式，
    并生成两个文件：
    1. Name_Sequences.txt (Enum_ID\t序列)
    2. Name_Sequences.fasta (FASTA格式)

    注意：原始文件中的 Enum_* 编号将被移除，只保留新编号和序列
    """
    os.makedirs(out_root, exist_ok=True)

    # 收集所有序列（只提取序列部分，移除原始 Enum_* 编号）
    all_sequences = []
    files = sorted(f for f in os.listdir(all_dir) if f.startswith("ALL_") and f.endswith(".txt"))

    for f in files:
        file_path = os.path.join(all_dir, f)
        with open(file_path, "r") as r:
            for line in r:
                line = line.strip()
                if line:
                    # 分割行，提取序列部分（跳过原始 Enum_* 编号）
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        # 第二部分及之后才是实际序列
                        sequence = "\t".join(parts[1:])
                        all_sequences.append(sequence)
                    elif len(parts) == 1:
                        # 如果只有序列（没有原始编号），直接使用
                        all_sequences.append(parts[0])

    print(f"[INFO] Total sequences collected: {len(all_sequences)}")

    # 重新编号并导出
    txt_out = os.path.join(out_root, "Name_Sequences.txt")
    fasta_out = os.path.join(out_root, "Name_Sequences.fasta")

    with open(txt_out, "w") as txt_file, open(fasta_out, "w") as fasta_file:
        for idx, seq in enumerate(all_sequences, start=1):
            enum_id = f"Enum_{idx:08d}"  # Enum_00000001 格式

            # 写入 Name_Sequences.txt (Tab分隔)
            txt_file.write(f"{enum_id}\t{seq}\n")

            # 写入 FASTA 格式
            fasta_file.write(f">{enum_id}\n{seq}\n")

    print(f"[INFO] Exported Name_Sequences.txt: {txt_out}")
    print(f"[INFO] Exported Name_Sequences.fasta: {fasta_out}")
    print(f"[INFO] Total sequences exported: {len(all_sequences)}")

    return len(all_sequences)


# =========================
# 主程序
# =========================
def main():

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--aa_lib", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--fresh", type=int, default=1)
    ap.add_argument("--clean", action="store_true", help="清理所有输出目录和进度文件，重新开始")

    args = ap.parse_args()

    model = args.model.upper()
    aa_lib = list(dict.fromkeys(args.aa_lib.upper()))

    workers = max(1, os.cpu_count() - 1)

    out_root = os.path.abspath(args.out_root)

    d1 = os.path.join(out_root, "01_step1")
    d2 = os.path.join(out_root, "02_step2")
    d3 = os.path.join(out_root, "03_step3")

    cand_dir = os.path.join(out_root, "Candidates")
    all_dir = os.path.join(out_root, "ALL_Candidates")
    export_dir = os.path.join(out_root, "Exported_Sequences")
    progress_file = os.path.join(out_root, "progress.json")

    # =========================
    # 🔥 清理模式：删除所有输出目录和进度文件，然后继续执行
    # =========================
    if args.clean:
        print("\n🧹 [CLEAN] 开始清理输出目录...")
        dirs_to_clean = [d1, d2, d3, cand_dir, all_dir, export_dir]
        files_to_clean = [progress_file]

        cleaned_count = 0
        for d in dirs_to_clean:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"  [CLEAN] 已删除目录: {d}")
                cleaned_count += 1

        for f in files_to_clean:
            if os.path.exists(f):
                os.remove(f)
                print(f"  [CLEAN] 已删除文件: {f}")
                cleaned_count += 1

        print(f"🧹 [CLEAN] 清理完成! 共清理 {cleaned_count} 个项目")
        print("\n🚀 [CLEAN] 清理完成，继续执行主流程...")
        # 清理后强制 fresh=1，确保从头开始
        args.fresh = 1

    os.makedirs(out_root, exist_ok=True)

    if args.fresh:
        if os.path.exists(progress_file):
            os.remove(progress_file)

    prog = load_progress(progress_file)

    fix_pos, remain = auto_split_star(model, aa_lib)

    print(f"[INFO] remain_star={remain}")
    print(f"[INFO] templates={len(aa_lib)**len(fix_pos)}")

    templates = generate_templates(model, aa_lib, fix_pos)

    for tpl in templates:

        if tpl in prog["done"]:
            print("[SKIP]", tpl)
            continue

        print("\n===== TEMPLATE =====", tpl)

        recreate_dir(d1)
        recreate_dir(d2)
        recreate_dir(d3)

        run(["python", "step1_enum_basic_filter.py",
             "--model", tpl,
             "--aa_lib", "".join(aa_lib),
             "--outdir", d1,
             "--workers", str(workers)])

        run(["python", "step2_face_fast_filter.py",
             "--indir", d1,
             "--outdir", d2,
             "--workers", str(workers)])

        run(["python", "step3_lowmic_filter.py",
             "--indir", d2,
             "--outdir", d3,
             "--workers", str(workers)])

        # =========================
        # 🔥 每个模板独立 Candidates
        # =========================
        tpl_name = tpl.replace("*", "")
        tpl_out = os.path.join(cand_dir, f"{tpl_name}_Candidates.txt")

        append_candidates(d3, tpl_out)

        prog["done"].append(tpl)
        save_progress(progress_file, prog)

    # =========================
    # 🔥 最终合并 ALL
    # =========================
    print("\n[MERGE] All Candidates...")
    merge_all_candidates(cand_dir, all_dir, chunk=1_000_000)

    # =========================
    # 🔥 重新编号 + 生成 Name Sequences 和 FASTA
    # =========================
    print("\n[RENUMBER] Re-numbering sequences and generating Name Sequences...")
    export_dir = os.path.join(out_root, "Exported_Sequences")
    total_count = renumber_and_export_sequences(all_dir, export_dir)
    print(f"[INFO] Exported {total_count} unique sequences with Enum_* IDs")

    print("\n✅ ALL DONE")


if __name__ == "__main__":
    main()