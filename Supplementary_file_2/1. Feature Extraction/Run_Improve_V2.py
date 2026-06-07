#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run_Improve_V2.py - 修订版特征计算脚本

功能：
1. 计算基本特征（Length, Hydro, HMom, z, pI, FreqPolar, FreqNonPolar, Freq_A-Y, HydrophobicFace, HydrophilicFace等）
   - 来源: PeptidePipeline_Core_V3.py 和 amp_features_basic_V2.py
2. 计算HCS特征（HMom_HoF, HMom_HoF_HELNET, HCS4_HoF_Mean, HCS4_HoF_Mean_SD, HCS3_HoF_Mean, HCS3_HoF_Mean_SD等）
   - 来源: HCS_Dynamic.py
3. 输出xlsx格式
python Run_Improve_V2.py Q1_19_Seq.txt Feature.txt --out-root Results/Q1_19_Results
输入: Q1_19_Seq.txt (Name, Sequences, Exp_Log2_MIC)
输出: xlsx文件
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# 导入特征计算模块
from amp_features_basic_V2 import compute_basic_features
import HCS_Dynamic as hcs

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def load_feature_order(feature_txt: Path) -> List[str]:
    """读取Feature.txt获取特征顺序"""
    lines = feature_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    order = [l.strip() for l in lines if l.strip()]
    if not order:
        raise RuntimeError("Feature.txt 为空")
    return order


def detect_delimiter(path: Path) -> str:
    """检测文件分隔符"""
    first_line = path.read_text(errors="ignore").splitlines()[0]
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    return "\t"  # 默认tab分隔


def read_table_file(path: Path) -> List[tuple]:
    """读取表格文件（支持tab/csv格式）"""
    delimiter = detect_delimiter(path)
    df = pd.read_csv(path, sep=delimiter)
    items = []

    for i, row in df.iterrows():
        first = str(row.iloc[0]).strip()
        second = str(row.iloc[1]).strip().upper()

        if all(c in AA_SET for c in second):
            items.append((first, second))
        else:
            first_upper = first.upper()
            if all(c in AA_SET for c in first_upper):
                items.append((f"{path.stem}_{i+1}", first_upper))

    return items


def read_fasta(path: Path) -> List[tuple]:
    """读取FASTA文件"""
    items = []
    name = None
    seq_parts = []

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name and seq_parts:
                seq = "".join(seq_parts).upper()
                if all(c in AA_SET for c in seq):
                    items.append((name, seq))
            name = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)

    if name and seq_parts:
        seq = "".join(seq_parts).upper()
        if all(c in AA_SET for c in seq):
            items.append((name, seq))

    return items


def collect_sequences(input_path: Path) -> List[tuple]:
    """收集输入文件中的所有序列"""
    items = []

    if input_path.is_file():
        if input_path.suffix.lower() in [".fa", ".fasta", ".faa"]:
            return read_fasta(input_path)
        else:
            return read_table_file(input_path)

    for p in input_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in [".fa", ".fasta", ".faa"]:
            items.extend(read_fasta(p))
        elif p.suffix.lower() in [".txt", ".csv", ".tsv"]:
            items.extend(read_table_file(p))

    return items


def _to_float(x, default=0.0) -> float:
    """安全转换为浮点数"""
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def compute_one(name: str, seq: str, exp_mic: str = "") -> Dict[str, Any]:
    """
    计算单个肽段的所有特征

    Args:
        name: 序列名称
        seq: 氨基酸序列
        exp_mic: 实验Log2_MIC值

    Returns:
        包含所有特征的字典
    """
    # 1. 计算基本特征（来自amp_features_basic_V2.py）
    feats = compute_basic_features(name, seq)

    # 添加Exp_Log2_MIC列
    feats["Exp_Log2_MIC"] = exp_mic

    # 2. 计算HCS特征（来自HCS_Dynamic.py）
    # HCS_Dynamic.py包含HMom_HoF和HMom_HoF_HELNET（替代原HMom_HoF）
    hcs_feats = hcs.calculate_hcs_features(seq, name)

    # 提取需要的HCS特征
    # 注意：HMom_HoF被替换为HMom_HoF和HMom_HoF_HELNET
    hcs_keys = [
        'HMom_HoF',
        'HMom_HoF_HELNET',
        'HCS4_HoF_Mean',
        'HCS4_HoF_Mean_SD',
        'HCS3_HoF_Mean',
        'HCS3_HoF_Mean_SD',
        'HCS4_HoF_1',
        'HCS4_HoF_2',
        'HCS3_HoF_1',
        'HCS3_HoF_2',
        'HCS_Pair1',
        'HCS_Pair2',
        'HCS_Pair3'
    ]

    for key in hcs_keys:
        feats[key] = hcs_feats.get(key, 0.0)

    return feats


def process_sequences(input_path: Path, feature_txt: Path, output_path: Path):
    """
    处理所有序列并输出xlsx文件

    Args:
        input_path: 输入文件路径
        feature_txt: Feature.txt路径
        output_path: 输出xlsx文件路径
    """
    feature_order = load_feature_order(feature_txt)
    items = collect_sequences(input_path)

    if not items:
        raise RuntimeError("No valid sequences found.")

    print(f"Found {len(items)} valid sequences")

    # 计算每条序列的特征
    rows = []
    for i, (name, seq) in enumerate(items):
        if i % 10 == 0:
            print(f"Processing sequence {i+1}/{len(items)}...")

        # 从原始数据中获取Exp_Log2_MIC
        exp_mic = ""
        if len(items) > 0:
            # 尝试从原始文件读取Exp_Log2_MIC
            try:
                delimiter = detect_delimiter(input_path)
                df = pd.read_csv(input_path, sep=delimiter)
                if len(df) > i:
                    if 'Exp_Log2_MIC' in df.columns:
                        exp_mic = df.iloc[i]['Exp_Log2_MIC']
            except Exception:
                pass

        feats = compute_one(name, seq, str(exp_mic) if pd.notna(exp_mic) else "")
        rows.append(feats)

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 确保所有Feature.txt中的列都存在
    for col in feature_order:
        if col not in df.columns:
            # 根据列名类型设置默认值
            if "Freq" in col or "HMom" in col or col.startswith("HCS"):
                df[col] = 0.0
            else:
                df[col] = ""

    # 按照Feature.txt的顺序排列列 - 仅输出Feature.txt中定义的列
    # 过滤掉所有不在Feature.txt中的列
    final_cols = [col for col in feature_order if col in df.columns]

    # 对于缺失的列添加默认值
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0  # 默认值

    df = df[final_cols]

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为xlsx格式
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"[Saved] {output_path}")
    print(f"[Info] Total columns: {len(final_cols)}, Total rows: {len(df)}")


def main():
    """主函数"""
    if "--input" in sys.argv:
        parser = argparse.ArgumentParser(description='Run_Improve_V2.py - 特征计算脚本')
        parser.add_argument("--input", required=True, help="输入文件路径")
        parser.add_argument("--feature", default="Feature.txt", help="Feature.txt路径")
        parser.add_argument("--out-root", help="输出文件路径（不含扩展名）")
        args = parser.parse_args()

        input_path = Path(args.input)
        feature_txt = Path(args.feature)

        if args.out_root:
            output_path = Path(args.out_root).with_suffix(".xlsx")
        else:
            output_path = Path.cwd() / f"Results/{input_path.stem}_Features.xlsx"

        process_sequences(input_path, feature_txt, output_path)
    else:
        if len(sys.argv) < 2:
            print("Usage: python Run_Improve_V2.py input.txt [feature.txt]", file=sys.stderr)
            sys.exit(1)

        input_path = Path(sys.argv[1])
        feature_txt = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Feature.txt")
        output_path = Path.cwd() / f"Results/{input_path.stem}_Features.xlsx"

        process_sequences(input_path, feature_txt, output_path)


if __name__ == "__main__":
    main()