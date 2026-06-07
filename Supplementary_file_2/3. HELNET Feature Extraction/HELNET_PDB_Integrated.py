#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HELNET_PDB_Integrated.py - 完整整合版本
基于 HELNET_PDB_Core_Mark.py，包含 PeptidePipeline_Core_V3 和 HELNET_PDB_Core_V5 的所有代码。
支持多进程处理 PDB，生成 JSON/TXT、螺旋轮图、完整摘要 Excel 和模型数据 Excel。

用法：
  python HELNET_PDB_Integrated.py --seq_file Q1_19_Seq.txt --pdb_dir structures --output_dir Q1_19_Seq_HELNET_PDB_Mark --clean --txt --HELNET
"""

import os
import sys
import re
import json
import shutil
import argparse
import math
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from Bio.PDB import PDBParser, is_aa

# ============================ 以下为 PeptidePipeline_Core_V3 完整代码 ============================

# Fauchère-Pliska hydrophobicity scale (1983)
FAUCHERE_PLISKA = {
    'A': 0.310, 'R': -1.010, 'N': -0.600, 'D': -0.770, 'C': 1.540,
    'Q': -0.220, 'E': -0.640, 'G': 0.000, 'H': 0.130, 'I': 1.800,
    'L': 1.700, 'K': -0.990, 'M': 1.230, 'F': 1.790, 'P': 0.720,
    'S': -0.040, 'T': 0.260, 'W': 2.250, 'Y': 0.960, 'V': 1.220
}

# Amino acid classifications
HYDROPHOBIC_FACE_RESIDUES = {'A', 'L', 'I', 'V', 'M', 'P', 'F', 'W', 'Y'}
HYDROPHILIC_FACE_RESIDUES = {'A', 'D', 'E', 'G', 'H', 'K', 'N', 'Q', 'R', 'S', 'T'}
CHARGED_RESIDUES = {'D', 'E', 'R', 'K'}
POLAR_RESIDUES = {'E', 'D', 'K', 'R', 'S', 'T', 'N', 'Q', 'H', 'G'}
NON_POLAR_RESIDUES = {'A', 'C', 'I', 'L', 'M', 'F', 'P', 'V', 'W', 'Y'}

ALL_AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]

AA_ORDER = ["A","C","D","E","F",
            "G","H","I","K","L",
            "M","N","P","Q","R",
            "S","T","V","W","Y"]

# Helical wheel properties
ROUND_SIZE = 18
RESIDUES_PER_TURN = 3.6
DEGREES_PER_RESIDUE = 100.0
DEG = DEGREES_PER_RESIDUE

# Amino acid pKa values
AMINO_ACID_PKA = {
    'A': {'pKa_C': 2.34, 'pKa_N': 9.69, 'pKa_R': None},
    'R': {'pKa_C': 2.17, 'pKa_N': 9.04, 'pKa_R': 12.48},
    'N': {'pKa_C': 2.02, 'pKa_N': 8.80, 'pKa_R': None},
    'D': {'pKa_C': 1.88, 'pKa_N': 9.60, 'pKa_R': 3.65},
    'C': {'pKa_C': 1.96, 'pKa_N': 10.28, 'pKa_R': 8.18},
    'Q': {'pKa_C': 2.17, 'pKa_N': 9.13, 'pKa_R': None},
    'E': {'pKa_C': 2.19, 'pKa_N': 9.67, 'pKa_R': 4.25},
    'G': {'pKa_C': 2.34, 'pKa_N': 9.60, 'pKa_R': None},
    'H': {'pKa_C': 1.82, 'pKa_N': 9.17, 'pKa_R': 6.00},
    'I': {'pKa_C': 2.36, 'pKa_N': 9.68, 'pKa_R': None},
    'L': {'pKa_C': 2.36, 'pKa_N': 9.60, 'pKa_R': None},
    'K': {'pKa_C': 2.18, 'pKa_N': 8.95, 'pKa_R': 10.53},
    'M': {'pKa_C': 2.28, 'pKa_N': 9.21, 'pKa_R': None},
    'F': {'pKa_C': 1.83, 'pKa_N': 9.13, 'pKa_R': None},
    'P': {'pKa_C': 1.99, 'pKa_N': 10.60, 'pKa_R': None},
    'S': {'pKa_C': 2.21, 'pKa_N': 9.15, 'pKa_R': None},
    'T': {'pKa_C': 2.11, 'pKa_N': 9.10, 'pKa_R': None},
    'W': {'pKa_C': 2.38, 'pKa_N': 9.39, 'pKa_R': None},
    'Y': {'pKa_C': 2.20, 'pKa_N': 9.11, 'pKa_R': 10.07},
    'V': {'pKa_C': 2.32, 'pKa_N': 9.62, 'pKa_R': None}
}

DEFAULT_PKA_N_TERM = 8.0
DEFAULT_PKA_C_TERM = 3.1

# Weighted HoF parameters
D_OFFSETS = (3, 4, 7)
W_COEF = {3: 0.6, 4: 1.0, 7: 0.4}

# ---------------------------- Core Functions (from Core.py) ----------------------------
def calculate_aa_counts(sequence: str) -> Dict[str, int]:
    """统计20种标准氨基酸的个数"""
    counter = Counter(sequence)
    return {aa: counter.get(aa, 0) for aa in AA_ORDER}

def calculate_mean_hydrophobicity(sequence: str) -> float:
    """计算平均疏水性"""
    if not sequence:
        return 0.0
    total = 0.0
    valid_count = 0
    for aa in sequence:
        value = FAUCHERE_PLISKA.get(aa)
        if value is not None:
            total += value
            valid_count += 1
    return total / valid_count if valid_count > 0 else 0.0

def calculate_mean_amphipathic_moment(sequence: str) -> float:
    """计算平均两亲性力矩"""
    if not sequence:
        return 0.0
    delta_radians = math.radians(100.0)
    sum_sin = 0.0
    sum_cos = 0.0
    n = len(sequence)
    for i, aa in enumerate(sequence):
        hydrophobicity = FAUCHERE_PLISKA.get(aa, 0.0)
        if hydrophobicity is not None:
            angle = i * delta_radians
            sum_sin += hydrophobicity * math.sin(angle)
            sum_cos += hydrophobicity * math.cos(angle)
    return math.sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / n

def calculate_net_charge(sequence: str) -> float:
    """计算净电荷（HeliQuest方法，pH=7.4）"""
    if not sequence:
        return 0.0
    seq = sequence.upper()
    pos = seq.count('K') + seq.count('R')
    neg = seq.count('D') + seq.count('E')
    return float(pos - neg)

def calculate_polar_residues_ratio(sequence: str) -> float:
    """计算极性氨基酸比例"""
    if not sequence:
        return 0.0
    polar_count = 0
    for aa in sequence:
        if aa in POLAR_RESIDUES:
            polar_count += 1
    return polar_count / len(sequence) if sequence else 0.0

def calculate_non_polar_residues_ratio(sequence: str) -> float:
    """计算非极性氨基酸比例"""
    if not sequence:
        return 0.0
    non_polar_count = 0
    for aa in sequence:
        if aa in NON_POLAR_RESIDUES:
            non_polar_count += 1
    return non_polar_count / len(sequence) if sequence else 0.0

def calculate_charge_at_ph(sequence: str, pH: float) -> float:
    """计算特定pH下的电荷"""
    if not sequence:
        return 0.0
    n_term_aa = sequence[0]
    c_term_aa = sequence[-1]
    n_term_pKa = AMINO_ACID_PKA.get(n_term_aa, {}).get('pKa_N', DEFAULT_PKA_N_TERM)
    c_term_pKa = AMINO_ACID_PKA.get(c_term_aa, {}).get('pKa_C', DEFAULT_PKA_C_TERM)
    n_term_charge = 1.0 / (1.0 + 10**(pH - n_term_pKa))
    c_term_charge = -1.0 / (1.0 + 10**(c_term_pKa - pH))
    side_chain_charge = 0.0
    for aa in sequence:
        pKa_data = AMINO_ACID_PKA.get(aa, {})
        side_chain_pKa = pKa_data.get('pKa_R')
        if side_chain_pKa is not None:
            if aa in ['D', 'E', 'C', 'Y']:
                side_chain_charge -= 1.0 / (1.0 + 10**(side_chain_pKa - pH))
            else:
                side_chain_charge += 1.0 / (1.0 + 10**(pH - side_chain_pKa))
    total_charge = n_term_charge + c_term_charge + side_chain_charge
    return total_charge

def calculate_isoelectric_point(sequence: str, tolerance: float = 0.01, max_iterations: int = 100) -> float:
    """计算等电点 (pI)"""
    if not sequence:
        return 7.0
    pH_min = 0.0
    pH_max = 14.0
    for _ in range(max_iterations):
        pH_mid = (pH_min + pH_max) / 2.0
        charge = calculate_charge_at_ph(sequence, pH_mid)
        if abs(charge) < tolerance:
            return pH_mid
        if charge > 0:
            pH_min = pH_mid
        else:
            pH_max = pH_mid
    return (pH_min + pH_max) / 2.0

def calculate_d_value(h_mom: float, z: float) -> float:
    """计算D值"""
    return 0.944 * h_mom + 0.33 * z

def determine_helix_type(d_value: float) -> str:
    """确定螺旋类型"""
    if d_value < 0.68:
        return "Helix"
    elif d_value <= 1.34:
        return "Possible Lipid-Binding Helix"
    else:
        return "Lipid-Binding Helix"

# ---------------------------- Multi-Round Helical Wheel Functions ----------------------------
def get_round_sequences(sequence: str) -> List[str]:
    """将序列分割为轮次：第1轮(1-18)、第2轮(19-36)、第3轮(37-54)"""
    rounds = []
    length = len(sequence)
    for i in range(3):
        start = i * ROUND_SIZE
        end = min((i + 1) * ROUND_SIZE, length)
        if start < length:
            rounds.append(sequence[start:end])
        else:
            rounds.append("")
    return rounds

def generate_helical_wheel_mapping(sequence_length: int, start_index: int = 0) -> List[int]:
    """生成螺旋轮映射（按角度排序的 index 序列）"""
    if sequence_length <= 0:
        return []
    angles = []
    for i in range(sequence_length):
        angle = (start_index + i) * DEGREES_PER_RESIDUE
        angle = angle % 360.0
        angles.append((i, angle))
    angles.sort(key=lambda x: x[1])
    mapping = [idx for idx, _ in angles]
    return mapping

def generate_helical_wheel_sequence_optimized(sequence: str, start_index: int = 0) -> str:
    """生成螺旋轮序列（按角度排序后的 residue 串）"""
    if not sequence:
        return ""
    sequence_length = len(sequence)
    mapping = generate_helical_wheel_mapping(sequence_length, start_index)
    return ''.join(sequence[idx] for idx in mapping)

def is_hydrophobic_amino_acid(aa: str, context_round: int = 1, left_aa: str = None, right_aa: str = None) -> bool:
    """判断氨基酸是否为疏水性氨基酸（保持原 Core_V2 规则）"""
    if aa in HYDROPHOBIC_FACE_RESIDUES:
        return True
    elif aa == 'G':
        if context_round >= 2:
            return True
        elif context_round == 1:
            return (left_aa in HYDROPHOBIC_FACE_RESIDUES and right_aa in HYDROPHOBIC_FACE_RESIDUES)
    return False

def find_hydrophobic_face_multiround(sequence: str) -> Tuple[List[int], int, str, List[int]]:
    """查找疏水面（保持原 Core_V2 规则）"""
    length = len(sequence)
    if length < ROUND_SIZE:
        return [], -1, "", []

    rounds = get_round_sequences(sequence)
    round1_seq = rounds[0]
    round2_seq = rounds[1]
    round3_seq = rounds[2]
    round1_length = len(round1_seq)

    mapping = generate_helical_wheel_mapping(round1_length, start_index=0)
    round1_wheel_seq = ''.join(round1_seq[idx] for idx in mapping)

    best_face = []
    best_start = -1
    best_face_sequence = ""
    best_face_linear_idxs: List[int] = []

    for start in range(round1_length):
        current_face = []
        current_face_seq: List[str] = []
        current_face_linear_idxs: List[int] = []

        for i in range(round1_length):
            pos = (start + i) % round1_length
            aa1 = round1_wheel_seq[pos]

            left_pos = (pos - 1 + round1_length) % round1_length
            right_pos = (pos + 1) % round1_length
            left_aa = round1_wheel_seq[left_pos]
            right_aa = round1_wheel_seq[right_pos]
            is_hydrophobic_round1 = is_hydrophobic_amino_acid(aa1, 1, left_aa, right_aa)

            round2_pos_exists = len(round2_seq) > pos
            if round2_pos_exists:
                aa2 = round2_seq[pos]
                if not is_hydrophobic_amino_acid(aa2, 2):
                    is_hydrophobic_round1 = False

            round3_pos_exists = len(round3_seq) > pos
            if round3_pos_exists:
                aa3 = round3_seq[pos]
                if not is_hydrophobic_amino_acid(aa3, 3):
                    is_hydrophobic_round1 = False

            if is_hydrophobic_round1:
                current_face.append(pos)
                current_face_seq.append(aa1)
                current_face_linear_idxs.append(mapping[pos])

                if round2_pos_exists:
                    aa2 = round2_seq[pos]
                    if is_hydrophobic_amino_acid(aa2, 2):
                        current_face_seq.append(aa2)
                        idx2 = round1_length + pos
                        if 0 <= idx2 < length:
                            current_face_linear_idxs.append(idx2)

                if round3_pos_exists:
                    aa3 = round3_seq[pos]
                    if is_hydrophobic_amino_acid(aa3, 3):
                        current_face_seq.append(aa3)
                        idx3 = round1_length + len(round2_seq) + pos
                        if 0 <= idx3 < length:
                            current_face_linear_idxs.append(idx3)

            else:
                if len(current_face) >= 5 and len(current_face) > len(best_face):
                    best_face = current_face.copy()
                    best_start = start
                    best_face_sequence = ''.join(current_face_seq)
                    best_face_linear_idxs = current_face_linear_idxs.copy()
                current_face = []
                current_face_seq = []
                current_face_linear_idxs = []

        if len(current_face) >= 5 and len(current_face) > len(best_face):
            best_face = current_face.copy()
            best_start = start
            best_face_sequence = ''.join(current_face_seq)
            best_face_linear_idxs = current_face_linear_idxs.copy()

    return best_face, best_start, best_face_sequence, best_face_linear_idxs

def find_hydrophilic_face_multiround(sequence: str, hydrophobic_face: List[int]) -> List[int]:
    """查找亲水面（保持原 Core_V2 规则）"""
    if not hydrophobic_face:
        return []

    hydrophobic_size = len(hydrophobic_face)
    rounds = get_round_sequences(sequence)
    round1_seq = rounds[0]
    round1_length = len(round1_seq)

    if hydrophobic_size > round1_length // 2:
        return []

    hydrophilic_size = max(1, hydrophobic_size - 2)
    if hydrophilic_size <= 0:
        return []

    round1_wheel_seq = generate_helical_wheel_sequence_optimized(round1_seq, start_index=0)
    hydrophilic_face = []
    center_index = hydrophobic_face[hydrophobic_size // 2]
    opposite_center = (center_index + round1_length // 2) % round1_length
    start_offset = -(hydrophilic_size // 2)

    if hydrophilic_size % 2 == 0:
        for i in range(start_offset, hydrophilic_size + start_offset):
            opposite_index = (opposite_center + i + round1_length) % round1_length
            hydrophilic_face.append(opposite_index)
    else:
        for i in range(start_offset, -start_offset + 1):
            opposite_index = (opposite_center + i + round1_length) % round1_length
            hydrophilic_face.append(opposite_index)

    for pos in hydrophilic_face:
        aa = round1_wheel_seq[pos]
        if aa not in HYDROPHILIC_FACE_RESIDUES:
            return []

    return hydrophilic_face

def find_helical_faces_multiround(sequence: str) -> Tuple[str, str, List[int], List[int], List[int], List[int]]:
    """查找螺旋面（保持原 Core_V2 规则）"""
    length = len(sequence)
    if length < ROUND_SIZE:
        return "None", "None", [], [], [], []

    hyd_positions_wheel, _, hydrophobic_face_seq, hyd_positions_linear = find_hydrophobic_face_multiround(sequence)
    if not hyd_positions_wheel:
        return "None", "None", [], [], [], []

    phil_positions_wheel = find_hydrophilic_face_multiround(sequence, hyd_positions_wheel)

    rounds = get_round_sequences(sequence)
    round1_seq = rounds[0]
    round1_length = len(round1_seq)
    mapping = generate_helical_wheel_mapping(round1_length, start_index=0)
    round1_wheel_seq = ''.join(round1_seq[idx] for idx in mapping)

    hydrophilic_seq = ''.join(round1_wheel_seq[pos] for pos in phil_positions_wheel) if phil_positions_wheel else "None"
    phil_positions_linear = [mapping[pos] for pos in phil_positions_wheel] if phil_positions_wheel else []

    return hydrophobic_face_seq, hydrophilic_seq, hyd_positions_wheel, phil_positions_wheel, hyd_positions_linear, phil_positions_linear

def calculate_hydrophobic_face_dispersion(face_positions: List[int], sequence_length: int) -> float:
    """计算疏水面在螺旋轮上的角度离散度（0~1）"""
    if not face_positions:
        return 0.0
    angles = [math.radians(pos * DEGREES_PER_RESIDUE) for pos in face_positions]
    sum_cos = sum(math.cos(a) for a in angles)
    sum_sin = sum(math.sin(a) for a in angles)
    R = math.sqrt(sum_cos**2 + sum_sin**2) / len(angles)
    return 1.0 - R

def weighted_hf_score(seq: str, positions: List[int]) -> float:
    """计算加权HoF分数"""
    if not positions or not seq:
        return 0.0
    pos_set = set(positions)
    score = 0.0
    L = len(seq)
    for i in pos_set:
        hi = FAUCHERE_PLISKA.get(seq[i], 0.0)
        for d in D_OFFSETS:
            j = i + d
            if j in pos_set and j < L:
                score += W_COEF[d] * hi * FAUCHERE_PLISKA.get(seq[j], 0.0)
    return score

def get_hydrophobic_face_positions_weighted(seq: str, threshold: float = 0.0) -> List[int]:
    """获取疏水面位置（基于加权HoF方法）"""
    return [i for i, aa in enumerate(seq) if FAUCHERE_PLISKA.get(aa, 0.0) > threshold]

def get_hydrophilic_face_positions_weighted(seq: str, threshold: float = 0.0) -> List[int]:
    """获取亲水面位置（基于加权HoF方法）"""
    return [i for i, aa in enumerate(seq) if FAUCHERE_PLISKA.get(aa, 0.0) <= threshold]

def calculate_amino_acid_freq_in_face(face_sequence: str, amino_acids: set) -> float:
    """计算特定氨基酸在面中的频率"""
    if not face_sequence or face_sequence == "None":
        return 0.0
    total_count = len(face_sequence)
    if total_count == 0:
        return 0.0
    count = sum(1 for aa in face_sequence if aa in amino_acids)
    return count / total_count

def calculate_continuity_ratio(face_sequence: str) -> float:
    """计算面中连续相同氨基酸的最大长度占比"""
    if not face_sequence or face_sequence == "None":
        return 0.0
    total_length = len(face_sequence)
    if total_length == 0:
        return 0.0
    max_continuity = 1
    current_continuity = 1
    for i in range(1, total_length):
        if face_sequence[i] == face_sequence[i-1]:
            current_continuity += 1
            if current_continuity > max_continuity:
                max_continuity = current_continuity
        else:
            current_continuity = 1
    return max_continuity / total_length

def calculate_all_face_frequency_features(hydrophobic_face_sequence: str, hydrophilic_face_sequence: str) -> Dict[str, float]:
    """计算所有面频率特征"""
    features = {}
    hydrophobic_amino_acids = {'A', 'L', 'I', 'V', 'M', 'P', 'F', 'W', 'Y', 'G'}
    for aa in hydrophobic_amino_acids:
        features[f'freq_{aa.lower()}_in_hydrophobic_face'] = calculate_amino_acid_freq_in_face(hydrophobic_face_sequence, {aa})
    for aa in ['K', 'R']:
        features[f'freq_{aa.lower()}_in_hydrophilic_face'] = calculate_amino_acid_freq_in_face(hydrophilic_face_sequence, {aa})
    features['hydrophobic_face_continuity_ratio'] = calculate_continuity_ratio(hydrophobic_face_sequence)
    features['hydrophilic_face_continuity_ratio'] = calculate_continuity_ratio(hydrophilic_face_sequence)
    return features

def wheel_corr_0_vs_100(seq: str) -> float:
    """计算螺旋轮0°和100°角度的相关性"""
    v0, v100 = [], []
    for i, a in enumerate(seq):
        h = FAUCHERE_PLISKA.get(a, 0.0)
        ang = math.radians(i * DEG)
        v0.append(h * math.cos(ang))
        v100.append(h * math.cos(ang + math.radians(100)))
    if np.std(v0) < 1e-6 or np.std(v100) < 1e-6:
        return 1.0
    return pearsonr(v0, v100)[0]

def find_faces_core(seq: str) -> Tuple[List[int], List[int]]:
    """核心面查找函数（基于疏水性正负）"""
    hyd, phil = [], []
    for i, a in enumerate(seq):
        (hyd if FAUCHERE_PLISKA.get(a, 0.0) > 0 else phil).append(i)
    return hyd, phil

def _compute_amp_features_safe(seq: str, hyd_positions: List[int], phil_positions: List[int]) -> Dict[str, float]:
    """调用 amp_face_features.py，输出 AMP_* 列（若依赖缺失则全 0）"""
    amp_defaults = {
        "AMP_muH_FP": 0.0,
        "AMP_HydrophobicFaceWidthDeg": 0.0,
        "AMP_HydrophilicFaceWidthDeg": 0.0,
        "AMP_HydrophobicFaceCenterDeg": 0.0,
        "AMP_HydrophilicFaceCenterDeg": 0.0,
        "AMP_FaceCenterSeparationDeg": 0.0,
        "AMP_FaceOppositionScore": 0.0,
        "AMP_PoreTopologyScore": 0.0,
        "AMP_HydrophobicContinuity": 0.0,
        "AMP_muQ_simple": 0.0,
        "AMP_phi_hydrophilic_DH": 0.0,
        "AMP_phi_hydrophobic_DH": 0.0,
        "AMP_AxialDipole_pz": 0.0,
        "AMP_PosStripeContinuity": 0.0,
        "AMP_PosStripeLongestRun": 0.0,
        "AMP_AromaticAnchorScore": 0.0,
        "AMP_HCS3_HoF": 0.0,
        "AMP_HCS4_HoF": 0.0,
        "AMP_Pairs1_HCS3_Score": 0.0,
        "AMP_Pairs1_HCS4_Score": 0.0,
        "AMP_Pairs2_HCS3_Score": 0.0,
        "AMP_Pairs2_HCS4_Score": 0.0,
    }
    try:
        from amp_face_features import compute_amp_face_features
    except Exception:
        return amp_defaults

    try:
        out = compute_amp_face_features(seq, hyd_positions, phil_positions)
        amp = dict(amp_defaults)
        amp["AMP_muH_FP"] = float(out.get("muH_FP", 0.0))
        amp["AMP_HydrophobicFaceWidthDeg"] = float(out.get("hydrophobic_face_width_deg", 0.0))
        amp["AMP_HydrophilicFaceWidthDeg"] = float(out.get("hydrophilic_face_width_deg", 0.0))
        amp["AMP_HydrophobicFaceCenterDeg"] = float(out.get("hydrophobic_face_center_deg", 0.0))
        amp["AMP_HydrophilicFaceCenterDeg"] = float(out.get("hydrophilic_face_center_deg", 0.0))
        amp["AMP_FaceCenterSeparationDeg"] = float(out.get("face_center_separation_deg", 0.0))
        amp["AMP_FaceOppositionScore"] = float(out.get("face_opposition_score", 0.0))
        amp["AMP_PoreTopologyScore"] = float(out.get("pore_topology_score", 0.0))
        amp["AMP_HydrophobicContinuity"] = float(out.get("hydrophobic_continuity", 0.0))
        amp["AMP_muQ_simple"] = float(out.get("muQ_simple", 0.0))
        amp["AMP_phi_hydrophilic_DH"] = float(out.get("phi_hydrophilic_DH", 0.0))
        amp["AMP_phi_hydrophobic_DH"] = float(out.get("phi_hydrophobic_DH", 0.0))
        amp["AMP_AxialDipole_pz"] = float(out.get("axial_dipole_charge_pz", out.get("axial_dipole_pz", 0.0)))
        amp["AMP_PosStripeContinuity"] = float(out.get("positive_stripe_continuity", out.get("pos_stripe_continuity", 0.0)))
        amp["AMP_PosStripeLongestRun"] = float(out.get("positive_stripe_longest_run", out.get("pos_stripe_longest_run", 0.0)))
        amp["AMP_AromaticAnchorScore"] = float(out.get("aromatic_anchor_score", 0.0))
        amp["AMP_HCS4_HoF"] = float(out.get("AMP_HCS4_HoF", out.get("HCS4_HoF", 0.0)))
        return amp
    except Exception:
        return amp_defaults

def compute_features(seq: str, name: str) -> Dict[str, Any]:
    """计算所有特征的函数"""
    L = len(seq)

    if L == 0:
        empty = {
            "Name": name,
            "Sequences": seq,
            "Length": 0,
            "Hyd": 0.0,
            "HMom": 0.0,
            "z": 0.0,
            "FreqPolar": 0.0,
            "FreqNonPolar": 0.0,
        }
        for aa in AA_ORDER:
            empty[f"Freq_{aa}"] = 0.0
        empty.update({
            "FreqW_in_HydrophobicFace": 0.0,
            "FreqA_in_HydrophobicFace": 0.0,
            "FreqL_in_HydrophobicFace": 0.0,
            "FreqI_in_HydrophobicFace": 0.0,
            "FreqV_in_HydrophobicFace": 0.0,
            "FreqM_in_HydrophobicFace": 0.0,
            "FreqP_in_HydrophobicFace": 0.0,
            "FreqF_in_HydrophobicFace": 0.0,
            "FreqY_in_HydrophobicFace": 0.0,
            "FreqG_in_HydrophobicFace": 0.0,
            "FreqK_in_HydrophilicFace": 0.0,
            "FreqR_in_HydrophilicFace": 0.0,
            "FreqContinuity_in_HydrophobicFace": 0.0,
            "FreqContinuity_in_HydrophilicFace": 0.0,
            "HydrophobicFace": "None",
            "HydrophilicFace": "None",
            "Hyd_of_hydrophobicFace": 0.0,
            "HMom_of_hydrophobicFace": 0.0,
            "Hyd_of_hydrophilicFace": 0.0,
            "HMom_of_hydrophilicFace": 0.0,
            "D_value": 0.0,
            "Helix_Type": "None",
            "pI": 7.0,
            "Charge_at_pH7": 0.0,
            "Weighted_HoF_Peptide": 0.0,
            "Weighted_HoF_HydrophobicFace": 0.0,
            "Weighted_HoF_HydrophilicFace": 0.0,
            "HydrophobicFaceLength": 0,
            "HydrophobicFaceDispersion": 0.0,
            "HydrophobicFaceSeqIdx": "",
            "HydrophilicFaceSeqIdx": "",
            "HydrophobicFaceSeq": "None",
            "HydrophilicFaceSeq": "None",
        })
        empty.update(_compute_amp_features_safe(seq, [], []))
        return empty

    hyd_mean = calculate_mean_hydrophobicity(seq)
    hmom = calculate_mean_amphipathic_moment(seq)
    z = calculate_net_charge(seq)

    aa_counts = calculate_aa_counts(seq)
    aa_freqs = {aa: (aa_counts[aa] / L if L > 0 else 0.0) for aa in AA_ORDER}

    freq_polar = calculate_polar_residues_ratio(seq)
    freq_non_polar = calculate_non_polar_residues_ratio(seq)

    hydrophobic_face_seq, hydrophilic_face_seq, hyd_positions, phil_positions, hyd_linear_idxs, phil_linear_idxs = find_helical_faces_multiround(seq)
    hydrophobic_face_seq_idx_str = ",".join(map(str, hyd_linear_idxs)) if hyd_linear_idxs else ""
    hydrophilic_face_seq_idx_str = ",".join(map(str, phil_linear_idxs)) if phil_linear_idxs else ""

    face_freq_features = calculate_all_face_frequency_features(hydrophobic_face_seq, hydrophilic_face_seq)

    hyd_hydrophobic_face = calculate_mean_hydrophobicity(hydrophobic_face_seq) if hydrophobic_face_seq != "None" else 0.0
    h_mom_hydrophobic_face = calculate_mean_amphipathic_moment(hydrophobic_face_seq) if hydrophobic_face_seq != "None" else 0.0
    hyd_hydrophilic_face = calculate_mean_hydrophobicity(hydrophilic_face_seq) if hydrophilic_face_seq != "None" else 0.0
    h_mom_hydrophilic_face = calculate_mean_amphipathic_moment(hydrophilic_face_seq) if hydrophilic_face_seq != "None" else 0.0

    hydrophobic_face_length = 0 if hydrophobic_face_seq == "None" else len(hydrophobic_face_seq)
    hydrophobic_face_dispersion = calculate_hydrophobic_face_dispersion(hyd_positions, L)

    d_value = calculate_d_value(hmom, z)
    helix_type = determine_helix_type(d_value)
    pI = calculate_isoelectric_point(seq)
    charge_at_pH7 = calculate_charge_at_ph(seq, 7.0)

    weighted_hf_peptide = weighted_hf_score(seq, list(range(L)))
    weighted_hyd_face_positions = list(hyd_linear_idxs or [])
    weighted_philic_face_positions = list(phil_linear_idxs or [])
    weighted_hf_hydrophobic_face = weighted_hf_score(seq, weighted_hyd_face_positions)
    weighted_hf_hydrophilic_face = weighted_hf_score(seq, weighted_philic_face_positions)

    features: Dict[str, Any] = {
        "Name": name,
        "Sequences": seq,
        "Length": L,
        "Hyd": hyd_mean,
        "HMom": hmom,
        "z": z,
        "FreqPolar": freq_polar,
        "FreqNonPolar": freq_non_polar,
    }

    for aa in AA_ORDER:
        features[f"Freq_{aa}"] = aa_freqs[aa]

    features.update({
        "FreqW_in_HydrophobicFace": face_freq_features.get('freq_w_in_hydrophobic_face', 0.0),
        "FreqA_in_HydrophobicFace": face_freq_features.get('freq_a_in_hydrophobic_face', 0.0),
        "FreqL_in_HydrophobicFace": face_freq_features.get('freq_l_in_hydrophobic_face', 0.0),
        "FreqI_in_HydrophobicFace": face_freq_features.get('freq_i_in_hydrophobic_face', 0.0),
        "FreqV_in_HydrophobicFace": face_freq_features.get('freq_v_in_hydrophobic_face', 0.0),
        "FreqM_in_HydrophobicFace": face_freq_features.get('freq_m_in_hydrophobic_face', 0.0),
        "FreqP_in_HydrophobicFace": face_freq_features.get('freq_p_in_hydrophobic_face', 0.0),
        "FreqF_in_HydrophobicFace": face_freq_features.get('freq_f_in_hydrophobic_face', 0.0),
        "FreqY_in_HydrophobicFace": face_freq_features.get('freq_y_in_hydrophobic_face', 0.0),
        "FreqG_in_HydrophobicFace": face_freq_features.get('freq_g_in_hydrophobic_face', 0.0),
        "FreqK_in_HydrophilicFace": face_freq_features.get('freq_k_in_hydrophilic_face', 0.0),
        "FreqR_in_HydrophilicFace": face_freq_features.get('freq_r_in_hydrophilic_face', 0.0),
        "FreqContinuity_in_HydrophobicFace": face_freq_features.get('hydrophobic_face_continuity_ratio', 0.0),
        "FreqContinuity_in_HydrophilicFace": face_freq_features.get('hydrophilic_face_continuity_ratio', 0.0),
        "HydrophobicFace": hydrophobic_face_seq,
        "HydrophilicFace": hydrophilic_face_seq,
        "Hyd_of_hydrophobicFace": hyd_hydrophobic_face,
        "HMom_of_hydrophobicFace": h_mom_hydrophobic_face,
        "Hyd_of_hydrophilicFace": hyd_hydrophilic_face,
        "HMom_of_hydrophilicFace": h_mom_hydrophilic_face,
        "D_value": d_value,
        "Helix_Type": helix_type,
        "pI": pI,
        "Charge_at_pH7": charge_at_pH7,
        "Weighted_HoF_Peptide": weighted_hf_peptide,
        "Weighted_HoF_HydrophobicFace": weighted_hf_hydrophobic_face,
        "Weighted_HoF_HydrophilicFace": weighted_hf_hydrophilic_face,
        "HydrophobicFaceLength": hydrophobic_face_length,
        "HydrophobicFaceDispersion": hydrophobic_face_dispersion,
        "HydrophobicFaceSeqIdx": hydrophobic_face_seq_idx_str,
        "HydrophilicFaceSeqIdx": hydrophilic_face_seq_idx_str,
        "HydrophobicFaceSeq": hydrophobic_face_seq,
        "HydrophilicFaceSeq": hydrophilic_face_seq,
    })

    _amp_round1_len = min(18, len(seq))
    hyd_positions_amp = [i for i in hyd_linear_idxs if 0 <= i < _amp_round1_len]
    phil_positions_amp = [i for i in phil_linear_idxs if 0 <= i < _amp_round1_len]
    features.update(_compute_amp_features_safe(seq, hyd_positions_amp, phil_positions_amp))

    return features

def generate_header() -> str:
    """生成表头行"""
    sample_features = compute_features("AAAAAAAAAAAAAAAAAA", "Sample")
    header_fields = list(sample_features.keys())
    return "\t".join(header_fields)

# ============================ 以下为 HELNET_PDB_Core_V5 完整代码 ============================

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

HYDROPHOBIC_SET = {'M', 'L', 'V', 'F', 'Y', 'W', 'I', 'A'}
HYDROPHILIC_SET = {'K', 'R'}

ANGLE_THRESHOLDS = {1: 120, 3: 70, 4: 50}

HYDROPHILICITY_SCALE = {'K': 1.0, 'R': 1.0}

SIDECHAIN_ENDPOINT = {
    'ALA': ('CB',), 'ARG': ('CZ',), 'ASN': ('CG', 'OD1', 'ND2'),
    'ASP': ('CG', 'OD1', 'OD2'), 'CYS': ('SG',), 'GLN': ('CD', 'OE1', 'NE2'),
    'GLU': ('CD', 'OE1', 'OE2'), 'GLY': ('CA',), 'HIS': ('CG', 'ND1', 'CE1', 'NE2', 'CD2'),
    'ILE': ('CD1',), 'LEU': ('CD1', 'CD2'), 'LYS': ('NZ',), 'MET': ('CE',),
    'PHE': ('CZ',), 'PRO': ('CG', 'CD'), 'SER': ('OG',), 'THR': ('OG1',),
    'TRP': ('CH2',), 'TYR': ('OH',), 'VAL': ('CG1', 'CG2'),
}

def compute_hydrophobic_moment(h_values: List[float], angles_deg: List[float]) -> float:
    N = len(h_values)
    if N == 0:
        return 0.0
    sum_cos = 0.0
    sum_sin = 0.0
    for h, ang in zip(h_values, angles_deg):
        rad = math.radians(ang)
        sum_cos += h * math.cos(rad)
        sum_sin += h * math.sin(rad)
    hm = math.sqrt(sum_cos**2 + sum_sin**2) / N
    return round(hm, 4)

def compute_moment_for_positions(positions: List[int],
                                 seq_residues: Dict[int, Dict],
                                 fauch_scale: Dict[str, float] = FAUCHERE_PLISKA) -> float:
    if not positions:
        return 0.0
    h_vals = []
    angles = []
    for pos in positions:
        if pos not in seq_residues:
            continue
        info = seq_residues[pos]
        aa = info['one_letter']
        ang = info['angle_deg']
        h_vals.append(fauch_scale.get(aa, 0.0))
        angles.append(ang)
    if len(h_vals) < 2:
        return 0.0
    return compute_hydrophobic_moment(h_vals, angles)

def find_hcs4_pairs(positions_set: Set[int]) -> List[Tuple[int, int]]:
    pairs = []
    for p in positions_set:
        if p + 4 in positions_set:
            pairs.append((p, p + 4))
    return sorted(pairs)

def find_hcs3_pairs(positions_set: Set[int]) -> List[Tuple[int, int]]:
    pairs = []
    for p in positions_set:
        if p + 3 in positions_set:
            pairs.append((p, p + 3))
    return sorted(pairs)

def find_hcs_triplets(positions_set: Set[int]) -> List[Tuple[int, int, int]]:
    triplets = set()
    for i in positions_set:
        if i + 3 in positions_set and i + 7 in positions_set:
            triplets.add(tuple(sorted((i, i+3, i+7))))
        if i + 4 in positions_set and i + 7 in positions_set:
            triplets.add(tuple(sorted((i, i+4, i+7))))
    return sorted(triplets)

def calculate_full_theoretical_moment(sequence: str) -> float:
    if not sequence:
        return 0.0
    h_vals = [FAUCHERE_PLISKA.get(aa, 0.0) for aa in sequence]
    angles = [i * 100.0 for i in range(len(sequence))]
    return compute_hydrophobic_moment(h_vals, angles)

def calc_hydrophobic_moment_actual(residues: List[Dict]) -> float:
    if not residues:
        return 0.0
    h_vals = [FAUCHERE_PLISKA.get(r['one_letter'], 0.0) for r in residues]
    angles = [r['angle_deg'] for r in residues]
    return compute_hydrophobic_moment(h_vals, angles)

def calc_hydrophobic_moment_theoretical(residues: List[Dict]) -> float:
    if not residues:
        return 0.0
    h_vals = [FAUCHERE_PLISKA.get(r['one_letter'], 0.0) for r in residues]
    angles = [(r['index'] - 1) * 100.0 for r in residues]
    return compute_hydrophobic_moment(h_vals, angles)

def calc_hydrophilic_moment_actual(residues: List[Dict]) -> float:
    if not residues:
        return 0.0
    h_vals = [HYDROPHILICITY_SCALE.get(r['one_letter'], 0.0) for r in residues]
    angles = [r['angle_deg'] for r in residues]
    return compute_hydrophobic_moment(h_vals, angles)

def get_sidechain_vector(residue):
    if not is_aa(residue):
        return None
    resname = residue.get_resname().upper()
    if resname == 'GLY':
        return None
    try:
        cb = residue['CB'].get_coord()
    except KeyError:
        return None
    endpoint_atoms = SIDECHAIN_ENDPOINT.get(resname)
    if not endpoint_atoms:
        return None
    coords = []
    for atom_name in endpoint_atoms:
        try:
            coords.append(residue[atom_name].get_coord())
        except KeyError:
            continue
    if not coords:
        try:
            coords.append(residue['CG'].get_coord())
        except KeyError:
            pass
    if not coords:
        return None
    center = np.mean(coords, axis=0)
    return center - cb

def fit_helix_axis(ca_coords: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.array(ca_coords)
    center = np.mean(coords, axis=0)
    centered = coords - center
    _, _, Vt = np.linalg.svd(centered)
    axis = Vt[0]
    axis = axis / np.linalg.norm(axis)
    return axis, center

def project_to_plane(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    proj_len = np.dot(vector, axis)
    return vector - proj_len * axis

def angle_diff(angle1: float, angle2: float) -> float:
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff

def check_connectivity(pos1: int, pos2: int, angle1: float, angle2: float) -> bool:
    gap = abs(pos1 - pos2)
    if gap not in ANGLE_THRESHOLDS:
        return False
    threshold = ANGLE_THRESHOLDS[gap]
    return angle_diff(angle1, angle2) <= threshold

def find_clusters(residues: List[Dict]) -> List[List[Dict]]:
    n = len(residues)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            pos_i = residues[i]['index']
            pos_j = residues[j]['index']
            angle_i = residues[i]['angle_deg']
            angle_j = residues[j]['angle_deg']
            if check_connectivity(pos_i, pos_j, angle_i, angle_j):
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            cluster = []
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                cluster.append(residues[node])
                for nb in adj[node]:
                    if not visited[nb]:
                        stack.append(nb)
            if cluster:
                clusters.append(cluster)
    return clusters

def min_arc_covering_points(angles, coverage_ratio=0.95):
    if not angles:
        return 0.0, 0.0, 0.0
    angles = np.sort(angles)
    n = len(angles)
    k = max(1, int(np.ceil(n * coverage_ratio)))
    if k == 1:
        return angles[0], angles[0], 0.0
    angles_ext = np.concatenate([angles, angles + 360])
    best_span = 360.0
    best_start = angles[0]
    best_end = angles[0]
    for i in range(n):
        window = angles_ext[i:i+k]
        span = window[-1] - window[0]
        if span < best_span:
            best_span = span
            best_start = window[0] % 360
            best_end = window[-1] % 360
    return best_start, best_end, best_span

def overlap_angle(start1, end1, start2, end2):
    def norm_angle(a):
        return a % 360
    s1, e1 = norm_angle(start1), norm_angle(end1)
    s2, e2 = norm_angle(start2), norm_angle(end2)
    def split_interval(s, e):
        if s <= e:
            return [(s, e)]
        else:
            return [(s, 360), (0, e)]
    intervals1 = split_interval(s1, e1)
    intervals2 = split_interval(s2, e2)
    total = 0.0
    for a1, b1 in intervals1:
        for a2, b2 in intervals2:
            lo = max(a1, a2)
            hi = min(b1, b2)
            if lo < hi:
                total += (hi - lo)
    return total

def compute_sidechain_angles(pdb_path: str) -> Dict:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('helix', pdb_path)
    model = structure[0]
    ca_coords = []
    residues_in_order = []
    for chain in model:
        for residue in chain:
            if not is_aa(residue):
                continue
            try:
                ca = residue['CA'].get_coord()
                ca_coords.append(ca)
                residues_in_order.append(residue)
            except KeyError:
                continue
    if len(ca_coords) < 3:
        raise ValueError(f"Not enough Cα atoms ({len(ca_coords)}) in {pdb_path}")
    axis, axis_center = fit_helix_axis(ca_coords)
    first_ca = ca_coords[0]
    last_ca = ca_coords[-1]
    diff = last_ca - first_ca
    if np.dot(diff, axis) < 0:
        axis = -axis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.9999:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    results = {}
    for idx, residue in enumerate(residues_in_order, start=1):
        resname = residue.get_resname().upper()
        one_letter = THREE_TO_ONE.get(resname, 'X')
        try:
            cb_coord = residue['CB'].get_coord()
        except KeyError:
            results[idx] = {'index': idx, 'resname': resname, 'one_letter': one_letter, 'error': 'Missing CB (Gly)'}
            continue
        side_vec = get_sidechain_vector(residue)
        if side_vec is None:
            results[idx] = {'index': idx, 'resname': resname, 'one_letter': one_letter, 'error': 'No sidechain endpoint atoms'}
            continue
        proj_vec = project_to_plane(side_vec, axis)
        proj_norm = np.linalg.norm(proj_vec)
        if proj_norm < 1e-6:
            results[idx] = {'index': idx, 'resname': resname, 'one_letter': one_letter, 'error': 'Projected vector too short'}
            continue
        proj_u = np.dot(proj_vec, u)
        proj_v = np.dot(proj_vec, v)
        angle_deg = np.degrees(np.arctan2(proj_v, proj_u)) % 360.0
        results[idx] = {
            'index': idx, 'resname': resname, 'one_letter': one_letter,
            'cb_coord': [float(x) for x in cb_coord],
            'sidechain_vector': [float(x) for x in side_vec],
            'projected_vector': [float(x) for x in proj_vec],
            'angle_deg': float(round(angle_deg, 2)),
            'projected_norm': float(round(proj_norm, 4)),
        }
    return {
        'pdb_file': os.path.basename(pdb_path),
        'axis_direction': [float(x) for x in axis],
        'axis_center': [float(x) for x in axis_center],
        'plane_basis_u': [float(x) for x in u],
        'plane_basis_v': [float(x) for x in v],
        'residues': results
    }

def analyze_surfaces(pdb_data: Dict) -> Dict:
    residues = pdb_data['residues']
    valid_residues = []
    for idx, info in residues.items():
        if 'angle_deg' in info:
            valid_residues.append({
                'index': info['index'],
                'one_letter': info['one_letter'],
                'angle_deg': info['angle_deg']
            })
    seq_residues_map = {r['index']: r for r in valid_residues}
    overall_actual = calc_hydrophobic_moment_actual(valid_residues)
    h_vectors, h_angles, p_vectors, p_angles = [], [], [], []
    for idx, info in pdb_data['residues'].items():
        if 'angle_deg' not in info:
            continue
        aa = info['one_letter']
        proj_vec = np.array(info.get('projected_vector', [0, 0, 0]))
        proj_norm = info.get('projected_norm', 0.0)
        if proj_norm < 1e-6:
            continue
        unit_vec = proj_vec / proj_norm
        angle = info['angle_deg']
        if aa in HYDROPHOBIC_SET:
            h_vectors.append(unit_vec); h_angles.append(angle)
        elif aa in HYDROPHILIC_SET:
            p_vectors.append(unit_vec); p_angles.append(angle)
    center_angle = None
    if h_vectors and p_vectors:
        h_sum = np.sum(h_vectors, axis=0); p_sum = np.sum(p_vectors, axis=0)
        norm_h, norm_p = np.linalg.norm(h_sum), np.linalg.norm(p_sum)
        if norm_h > 1e-6 and norm_p > 1e-6:
            cos_theta = np.dot(h_sum, p_sum) / (norm_h * norm_p)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            center_angle = float(np.degrees(np.arccos(cos_theta)))
    separation_index = None; span_h = span_p = overlap = 0.0
    if h_angles and p_angles:
        start_h, end_h, span_h = min_arc_covering_points(h_angles, 0.95)
        start_p, end_p, span_p = min_arc_covering_points(p_angles, 0.95)
        overlap = overlap_angle(start_h, end_h, start_p, end_p)
        total = span_h + span_p - overlap
        separation_index = 1.0 - (overlap / total) if total > 1e-6 else 0.0
    hydrophobic_residues = [r for r in valid_residues if r['one_letter'] in HYDROPHOBIC_SET]
    hydrophobic_clusters = find_clusters(hydrophobic_residues)
    hydrophilic_residues = [r for r in valid_residues if r['one_letter'] in HYDROPHILIC_SET]
    hydrophilic_clusters = find_clusters(hydrophilic_residues)
    hof_clusters = []
    for i, cluster in enumerate(hydrophobic_clusters, 1):
        positions = sorted([r['index'] for r in cluster])
        composition = ','.join(f"{r['one_letter']}{r['index']}" for r in sorted(cluster, key=lambda x: x['index']))
        hof_clusters.append({
            'id': i, 'positions': positions, 'composition': composition,
            'hmom_theoretical': calc_hydrophobic_moment_theoretical(cluster),
            'hmom_actual': calc_hydrophobic_moment_actual(cluster),
            'n_residues': len(cluster)
        })
    hif_clusters = []
    for i, cluster in enumerate(hydrophilic_clusters, 1):
        positions = sorted([r['index'] for r in cluster])
        composition = ','.join(f"{r['one_letter']}{r['index']}" for r in sorted(cluster, key=lambda x: x['index']))
        hif_clusters.append({
            'id': i, 'positions': positions, 'composition': composition,
            'n_residues': len(cluster),
            'hydrophilic_moment_actual': calc_hydrophilic_moment_actual(cluster)
        })
    hof_clusters_sorted = sorted(hof_clusters, key=lambda x: x['n_residues'], reverse=True)
    hif_clusters_sorted = sorted(hif_clusters, key=lambda x: x['n_residues'], reverse=True)
    largest_hof = hof_clusters_sorted[0] if hof_clusters_sorted else None
    hcs4_values, hcs3_values = [], []
    hcs_pair_values, hcs_pair_compositions = [], []
    if largest_hof and len(largest_hof['positions']) >= 2:
        pos_set = set(largest_hof['positions'])
        for p1, p2 in find_hcs4_pairs(pos_set):
            hcs4_values.append(compute_moment_for_positions([p1, p2], seq_residues_map))
        for p1, p2 in find_hcs3_pairs(pos_set):
            hcs3_values.append(compute_moment_for_positions([p1, p2], seq_residues_map))
        triplets = find_hcs_triplets(pos_set)
        for triplet in triplets:
            hm = compute_moment_for_positions(list(triplet), seq_residues_map)
            hcs_pair_values.append(round(hm, 4))
            comp = ','.join(f"{seq_residues_map[p]['one_letter']}{p}" for p in triplet)
            hcs_pair_compositions.append(comp)
    hcs4_mean = np.mean(hcs4_values) if hcs4_values else 0.0
    hcs4_sd = np.std(hcs4_values, ddof=1) if len(hcs4_values) > 1 else 0.0
    hcs3_mean = np.mean(hcs3_values) if hcs3_values else 0.0
    hcs3_sd = np.std(hcs3_values, ddof=1) if len(hcs3_values) > 1 else 0.0
    result = pdb_data.copy()
    result['hydrophobic_clusters'] = hof_clusters_sorted
    result['hydrophilic_clusters_sorted'] = hif_clusters_sorted
    result['largest_hydrophobic_cluster'] = largest_hof
    result['largest_hydrophilic_cluster'] = hif_clusters_sorted[0] if hif_clusters_sorted else None
    result['center_angle_deg'] = center_angle
    result['separation_index'] = separation_index
    result['hydrophobic_span_deg'] = span_h if h_angles else None
    result['hydrophilic_span_deg'] = span_p if p_angles else None
    result['overlap_deg'] = overlap if (h_angles and p_angles) else None
    result['overall_hmom_actual'] = overall_actual
    result['residue_count'] = len(valid_residues)
    result['hcs4_mean'] = round(hcs4_mean, 4)
    result['hcs4_sd'] = round(hcs4_sd, 4)
    result['hcs4_num'] = len(hcs4_values)
    result['hcs3_mean'] = round(hcs3_mean, 4)
    result['hcs3_sd'] = round(hcs3_sd, 4)
    result['hcs3_num'] = len(hcs3_values)
    result['hcs_pair_num'] = len(hcs_pair_values)
    result['hcs_pair_values'] = hcs_pair_values
    result['hcs_pair_compositions'] = hcs_pair_compositions
    return result

def save_results(data: Dict, out_json: str, out_txt: Optional[str] = None):
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  JSON saved: {out_json}")
    if out_txt:
        with open(out_txt, 'w') as f:
            f.write(f"PDB: {data['pdb_file']}\n")
            f.write(f"Helix axis direction: {data['axis_direction']}\n")
            f.write(f"Helix axis center: {data['axis_center']}\n")
            f.write(f"Plane basis u: {data['plane_basis_u']}\n")
            f.write(f"Plane basis v: {data['plane_basis_v']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Idx':>4} {'AA':>4} {'Angle(deg)':>12} {'ProjNorm':>10}\n")
            f.write("-" * 80 + "\n")
            for idx, info in data['residues'].items():
                if 'angle_deg' in info:
                    f.write(f"{idx:>4} {info['one_letter']:>4} {info['angle_deg']:>12.2f} {info['projected_norm']:>10.4f}\n")
                else:
                    f.write(f"{idx:>4} {info['one_letter']:>4} {'Error':>12} {info.get('error', '')}\n")
            f.write(f"\nValid residue count: {data.get('residue_count', 0)}\n")
            f.write(f"Full-length hydrophobic moment (actual): {data['overall_hmom_actual']:.4f}\n")
            if data.get('center_angle_deg') is not None:
                f.write(f"\nSpatial relationship:\n")
                f.write(f"  Center angle: {data['center_angle_deg']:.2f}°\n")
                f.write(f"  Separation Index: {data['separation_index']:.4f}\n")
                if data.get('hydrophobic_span_deg'):
                    f.write(f"  Hydrophobic 95% arc: {data['hydrophobic_span_deg']:.2f}°\n")
                    f.write(f"  Hydrophilic 95% arc: {data['hydrophilic_span_deg']:.2f}°\n")
                    f.write(f"  Overlap: {data['overlap_deg']:.2f}°\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("Hydrophobic clusters (HoF):\n")
            for clust in data.get('hydrophobic_clusters', []):
                f.write(f"  Cluster {clust['id']}: {clust['composition']}  |  HMom_theoretical = {clust['hmom_theoretical']:.4f}  |  HMom_actual = {clust['hmom_actual']:.4f}\n")
            f.write("\nHydrophilic clusters (HiF, K/R only):\n")
            for clust in data.get('hydrophilic_clusters_sorted', []):
                f.write(f"  Cluster {clust['id']}: {clust['composition']}  |  HiMom_actual = {clust['hydrophilic_moment_actual']:.4f}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("HCS indicators (based on largest hydrophobic cluster, using actual Cβ angles):\n")
            f.write(f"  HCS4_HoF_Mean_PDB: {data.get('hcs4_mean', 0.0):.4f}\n")
            f.write(f"  HCS4_HoF_Mean_PDB_SD: {data.get('hcs4_sd', 0.0):.4f}\n")
            f.write(f"  HCS4_HoF_Num_PDB: {data.get('hcs4_num', 0)}\n")
            f.write(f"  HCS3_HoF_Mean_PDB: {data.get('hcs3_mean', 0.0):.4f}\n")
            f.write(f"  HCS3_HoF_Mean_SD_PDB: {data.get('hcs3_sd', 0.0):.4f}\n")
            f.write(f"  HCS3_HoF_Num_PDB: {data.get('hcs3_num', 0)}\n")
            f.write(f"  HCS_Pair_Num_PDB: {data.get('hcs_pair_num', 0)}\n")
            pair_vals = data.get('hcs_pair_values', [])
            pair_comps = data.get('hcs_pair_compositions', [])
            for i, (val, comp) in enumerate(zip(pair_vals, pair_comps), 1):
                f.write(f"  HCS_Pair{i}_PDB_Num: {comp}\n")
                f.write(f"  HCS_Pair{i}_PDB: {val:.4f}\n")
        print(f"  TXT saved: {out_txt}")

# ============================ 以下为原 HELNET_PDB_Core_Mark.py 代码（含 main 及绘图功能） ============================

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Plotting will be disabled.")

# 颜色定义
HYDROPHOBIC = set("ILMFWYVP")
ACIDIC = set("DE")
BASIC = set("RK")
POLAR = set("NQSTYCMHK")
NONPOLAR = set("AGILMFPWV")

BG_COLORS = {
    'hydrophobic': (255, 255, 150),
    'hydrophilic': (173, 216, 230),
    'acidic': (255, 100, 100),
    'basic': (100, 149, 237),
    'polar': (144, 238, 144),
    'nonpolar': (255, 228, 196),
    'grey': (192, 192, 192),
    'default': (255, 255, 255),
}

def _normalize_color(color_tuple):
    return tuple(c / 255.0 for c in color_tuple)

def get_amino_acid_background_color(aa):
    aa = aa.upper()
    if aa in 'GA':
        return _normalize_color(BG_COLORS['grey'])
    elif aa in HYDROPHOBIC:
        return _normalize_color(BG_COLORS['hydrophobic'])
    elif aa in ACIDIC:
        return _normalize_color(BG_COLORS['acidic'])
    elif aa in BASIC:
        return _normalize_color(BG_COLORS['basic'])
    elif aa in POLAR:
        return _normalize_color(BG_COLORS['polar'])
    elif aa in NONPOLAR:
        return _normalize_color(BG_COLORS['nonpolar'])
    return _normalize_color(BG_COLORS['default'])

def draw_circle_with_label(ax, x, y, aa, position, diameter_cm=4.8, fontsize=90):
    radius = diameter_cm / 2.0
    bg_color = get_amino_acid_background_color(aa)
    circle = patches.Circle((x, y), radius, facecolor=bg_color, edgecolor='black',
                            linewidth=2, zorder=10)
    ax.add_patch(circle)
    label = f"{aa}{position}"
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', fontfamily='sans-serif', color='black', zorder=11)

def compute_plot_coordinates(sequence, v5_data, ref_angle):
    n = len(sequence)
    if n == 0:
        return []
    total_height_cm = (n + 1) * 4.8
    min_y = 0.0
    max_y = total_height_cm - 4.8
    coords = []
    for i, aa in enumerate(sequence):
        pos = i + 1
        res_info = v5_data['residues'].get(pos, {})
        angle = res_info.get('angle_deg', 0.0)
        if 'norm_z' in res_info:
            norm_z = res_info['norm_z']
        else:
            norm_z = i / (n - 1) if n > 1 else 0.0
        y = min_y + norm_z * (max_y - min_y)
        if angle > ref_angle:
            x = 16 - ((angle - ref_angle) * (16 / 360))
        else:
            x = (ref_angle - angle) * (16 / 360)
        coords.append((x, y, aa, pos, angle))
    return coords

def draw_helix_wheel(sequence, output_path, v5_data, ref_angle, angle_labels):
    if not _MATPLOTLIB_AVAILABLE:
        return False
    n = len(sequence)
    if n == 0:
        return False
    total_height_cm = (n + 1) * 4.8
    width_cm = 16.0
    fig_width_inches = 22
    fig_height_inches = fig_width_inches * (total_height_cm / width_cm)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width_inches, fig_height_inches))
    ax.set_xlim(-2.5, width_cm + 2.5)
    ax.set_ylim(-2.5, total_height_cm + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    border = patches.Rectangle((0, 0), width_cm, total_height_cm,
                               linewidth=12, edgecolor='black', facecolor='white')
    ax.add_patch(border)
    for xp in [4, 12]:
        ax.plot([xp, xp], [0, total_height_cm], linestyle='--', linewidth=12, color='black')
    x_positions = [0, 4, 8, 12, 16]
    for xp, ang in zip(x_positions, angle_labels):
        ax.text(xp, total_height_cm + 0.5, f"{ang}°", ha='center', va='bottom',
                fontsize=60, fontweight='bold')
    coords = compute_plot_coordinates(sequence, v5_data, ref_angle)
    for (x, y, aa, pos, _) in coords:
        draw_circle_with_label(ax, x, y, aa, pos, diameter_cm=4.8, fontsize=90)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.3, format='png')
    plt.close(fig)
    return True

def draw_helix_wheel_hof(sequence, output_path, v5_data):
    angle_labels = [150, 60, 330, 240, 150]
    return draw_helix_wheel(sequence, output_path, v5_data, 150, angle_labels)

def draw_helix_wheel_hif(sequence, output_path, v5_data):
    angle_labels = [330, 240, 150, 60, 330]
    return draw_helix_wheel(sequence, output_path, v5_data, 300, angle_labels)

def draw_helix_wheel_hof_hif(sequence, output_path, v5_data):
    angle_labels = [180, 90, 0, 270, 180]
    return draw_helix_wheel(sequence, output_path, v5_data, 180, angle_labels)

def draw_helix_wheel_with_mark(sequence, output_path, v5_data, positions,
                               ref_angle, angle_labels, box_color, label_prefix):
    if not _MATPLOTLIB_AVAILABLE:
        return False
    n = len(sequence)
    if n == 0:
        return False
    total_height_cm = (n + 1) * 4.8
    width_cm = 16.0
    fig_width_inches = 22
    fig_height_inches = fig_width_inches * (total_height_cm / width_cm)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width_inches, fig_height_inches))
    ax.set_xlim(-2.5, width_cm + 2.5)
    ax.set_ylim(-2.5, total_height_cm + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    border = patches.Rectangle((0, 0), width_cm, total_height_cm,
                               linewidth=12, edgecolor='black', facecolor='white')
    ax.add_patch(border)
    for xp in [4, 12]:
        ax.plot([xp, xp], [0, total_height_cm], linestyle='--', linewidth=12, color='black')
    x_positions = [0, 4, 8, 12, 16]
    for xp, ang in zip(x_positions, angle_labels):
        ax.text(xp, total_height_cm + 0.5, f"{ang}°", ha='center', va='bottom',
                fontsize=60, fontweight='bold')
    coords = compute_plot_coordinates(sequence, v5_data, ref_angle)
    for (x, y, aa, pos, _) in coords:
        draw_circle_with_label(ax, x, y, aa, pos, diameter_cm=4.8, fontsize=90)
    if positions:
        xs, ys = [], []
        for pos in positions:
            for (x, y, _, p, _) in coords:
                if p == pos:
                    xs.append(x); ys.append(y); break
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                     linewidth=8, edgecolor=box_color, facecolor='none', zorder=5)
            ax.add_patch(rect)
            ax.text(16, -4, f"{label_prefix}_Width: {x_max-x_min:.2f}", ha='right', va='top',
                    fontsize=40, fontweight='bold', color='black')
            ax.text(16, -6, f"{label_prefix}_Height: {y_max-y_min:.2f}", ha='right', va='top',
                    fontsize=40, fontweight='bold', color='black')
            ax.text(16, -8, f"{label_prefix}_Area: {(x_max-x_min)*(y_max-y_min):.2f}",
                    ha='right', va='top', fontsize=40, fontweight='bold', color='black')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.3, format='png')
    plt.close(fig)
    return True

def draw_helix_wheel_hof_with_mark(sequence, output_path, v5_data, hof_data):
    positions = hof_data.get('positions', []) if hof_data else []
    angle_labels = [150, 60, 330, 240, 150]
    return draw_helix_wheel_with_mark(sequence, output_path, v5_data, positions,
                                      150, angle_labels, 'red', 'HoF')

def draw_helix_wheel_hif_with_mark(sequence, output_path, v5_data, hif_data):
    positions = hif_data.get('positions', []) if hif_data else []
    angle_labels = [330, 240, 150, 60, 330]
    return draw_helix_wheel_with_mark(sequence, output_path, v5_data, positions,
                                      330, angle_labels, 'blue', 'HiF')

def compute_surface_metrics(sequence, v5_data, hof_positions, hif_positions):
    coords = compute_plot_coordinates(sequence, v5_data, 180)
    pos_to_xy = {p: (x, y) for (x, y, _, p, _) in coords}

    def get_bbox_and_center(positions):
        if not positions:
            return None, None, None, None, None
        x_vals = []
        y_vals = []
        for pos in positions:
            if pos in pos_to_xy:
                x, y = pos_to_xy[pos]
                x_vals.append(x)
                y_vals.append(y)
        if not x_vals:
            return None, None, None, None, None
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        center_x = x_min + width / 2
        center_y = y_min + height / 2
        return width, height, area, center_x, center_y

    hof_width, hof_height, hof_area, hof_cx, hof_cy = get_bbox_and_center(hof_positions)
    hif_width, hif_height, hif_area, hif_cx, hif_cy = get_bbox_and_center(hif_positions)

    distance = None
    delta_x = None
    delta_y = None
    if hof_cx is not None and hif_cx is not None:
        distance = math.hypot(hof_cx - hif_cx, hof_cy - hif_cy)
        delta_x = abs(hof_cx - hif_cx)
        delta_y = abs(hof_cy - hif_cy)

    return {
        "hof_width": hof_width if hof_width is not None else 0.0,
        "hof_height": hof_height if hof_height is not None else 0.0,
        "hof_area": hof_area if hof_area is not None else 0.0,
        "hif_width": hif_width if hif_width is not None else 0.0,
        "hif_height": hif_height if hif_height is not None else 0.0,
        "hif_area": hif_area if hif_area is not None else 0.0,
        "hof_center_x": hof_cx if hof_cx is not None else None,
        "hof_center_y": hof_cy if hof_cy is not None else None,
        "hif_center_x": hif_cx if hif_cx is not None else None,
        "hif_center_y": hif_cy if hif_cy is not None else None,
        "hof_hif_distance": distance if distance is not None else 0.0,
        "delta_x": delta_x if delta_x is not None else 0.0,
        "delta_y": delta_y if delta_y is not None else 0.0,
    }

def draw_helix_wheel_hof_hif_mark(sequence, output_path, v5_data,
                                  hof_positions, hif_positions):
    if not _MATPLOTLIB_AVAILABLE:
        return False
    n = len(sequence)
    if n == 0:
        return False

    total_height_cm = (n + 1) * 4.8
    width_cm = 16.0
    fig_width_inches = 22
    fig_height_inches = fig_width_inches * (total_height_cm / width_cm)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width_inches, fig_height_inches))
    ax.set_xlim(-2.5, width_cm + 2.5)
    ax.set_ylim(-2.5, total_height_cm + 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    border = patches.Rectangle((0, 0), width_cm, total_height_cm,
                               linewidth=12, edgecolor='black', facecolor='white')
    ax.add_patch(border)
    for xp in [4, 12]:
        ax.plot([xp, xp], [0, total_height_cm], linestyle='--', linewidth=12, color='black')
    ax.plot([8, 8], [0, total_height_cm], linestyle=':', linewidth=12, color='black')
    angle_labels = [180, 90, 0, 270, 180]
    x_positions = [0, 4, 8, 12, 16]
    for xp, ang in zip(x_positions, angle_labels):
        ax.text(xp, total_height_cm + 0.5, f"{ang}°", ha='center', va='bottom',
                fontsize=60, fontweight='bold')

    coords = compute_plot_coordinates(sequence, v5_data, 180)
    for (x, y, aa, pos, _) in coords:
        draw_circle_with_label(ax, x, y, aa, pos, diameter_cm=4.8, fontsize=90)

    hof_center = None
    if hof_positions:
        xs, ys = [], []
        for pos in hof_positions:
            for (x, y, _, p, _) in coords:
                if p == pos:
                    xs.append(x)
                    ys.append(y)
                    break
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=8, edgecolor='red', facecolor='none', zorder=5)
            ax.add_patch(rect)
            hof_center = (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2)
            ax.add_patch(patches.Circle(hof_center, 0.6, color='red', zorder=12))

    hif_center = None
    if hif_positions:
        xs, ys = [], []
        for pos in hif_positions:
            for (x, y, _, p, _) in coords:
                if p == pos:
                    xs.append(x)
                    ys.append(y)
                    break
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=8, edgecolor='blue', facecolor='none', zorder=5)
            ax.add_patch(rect)
            hif_center = (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2)
            ax.add_patch(patches.Circle(hif_center, 0.6, color='blue', zorder=12))

    if hof_center is not None and hif_center is not None:
        ax.plot([hof_center[0], hif_center[0]], [hof_center[1], hif_center[1]],
                linestyle='--', linewidth=4, color='black', zorder=4)
        dist = math.hypot(hof_center[0] - hif_center[0], hof_center[1] - hif_center[1])
        ax.text(16, -4, f"HoF_HiF_Distance: {dist:.2f}", ha='right', va='top',
                fontsize=40, fontweight='bold', color='black', zorder=11)

    hof_center_angle = v5_data.get('largest_hydrophobic_cluster', {}).get('center_angle')
    hif_center_angle = v5_data.get('largest_hydrophilic_cluster', {}).get('center_angle')
    if hof_center_angle is not None and hif_center_angle is not None:
        diff = abs(hof_center_angle - hif_center_angle)
        if diff > 180:
            diff = 360 - diff
        ax.text(16, -8, f"HoF-HiF Angle Diff: {diff:.1f}°", ha='right', va='top',
                fontsize=40, fontweight='bold', color='black', zorder=11)

    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0.3, format='png')
    plt.close(fig)
    return True

# ---------------------------- 特征计算辅助函数 ----------------------------
def compute_basic_features(sequence: str) -> dict:
    L = len(sequence)
    if L == 0:
        return {k: 0 for k in ("Length", "Hyd", "HMom", "z", "FreqPolar", "FreqNonPolar")}
    return {
        "Length": L,
        "Hyd": round(calculate_mean_hydrophobicity(sequence), 4),
        "HMom": round(calculate_mean_amphipathic_moment(sequence), 4),
        "z": round(calculate_net_charge(sequence), 4),
        "FreqPolar": round(calculate_polar_residues_ratio(sequence), 4),
        "FreqNonPolar": round(calculate_non_polar_residues_ratio(sequence), 4),
    }

def reconstruct_sequence_from_v5(analysis):
    residues = analysis['residues']
    max_pos = max([p for p in residues if isinstance(p, int)], default=0)
    return ''.join(residues.get(i, {}).get('one_letter', 'X') for i in range(1, max_pos+1))

def process_single_pdb(name, seq, exp, pdb_dir, output_dir, generate_txt, generate_plots):
    """Process one PDB file: compute, save, plot. Returns row dict for Model_Data."""
    basic = compute_basic_features(seq)
    pdb_path = os.path.join(pdb_dir, f"{name}.pdb")
    if not os.path.exists(pdb_path):
        print(f"  Warning: PDB not found for {name}: {pdb_path}")
        pdb_feat = {
            "HoF_HMom_PDB": "", "HMom_PDB": "", "HiF_Cluster_Count": 0, "HiF_HiMom_PDB": "",
            "Center_Angle_deg": "", "HCS4_HoF_Mean_PDB": "", "HCS4_HoF_Mean_SD_PDB": "",
            "HCS4_HoF_Num_PDB": 0, "HCS3_HoF_Mean_PDB": "", "HCS3_HoF_Mean_SD_PDB": "",
            "HCS3_HoF_Num_PDB": 0, "HCS_Pair_Num_PDB": 0,
            "HoF_HiF_Distance": "",
            "HoF_Width": "", "HoF_Height": "", "HoF_Area": "",
            "HiF_Width": "", "HiF_Height": "", "HiF_Area": "",
            "Delta_HoF_HiF_X": "", "Delta_HoF_HiF_Y": "",
            "_hcs_pair_values": [], "_hcs_pair_compositions": [], "_analysis": None
        }
    else:
        try:
            angles_data = compute_sidechain_angles(pdb_path)
            analysis = analyze_surfaces(angles_data)
            json_path = os.path.join(output_dir, f"{name}_sidechain.json")
            txt_path = os.path.join(output_dir, f"{name}_sidechain.txt") if generate_txt else None
            save_results(analysis, json_path, txt_path)
            lhc = analysis.get("largest_hydrophobic_cluster")
            hof_hmom = lhc.get("hmom_actual", "") if lhc else ""
            overall_hmom = analysis.get("overall_hmom_actual", "")
            hif_clusters = analysis.get("hydrophilic_clusters_sorted", [])
            hif_count = len(hif_clusters)
            largest_hif = hif_clusters[0] if hif_clusters else None
            hif_himom = largest_hif.get("hydrophilic_moment_actual", "") if largest_hif else ""
            center_angle = analysis.get("center_angle_deg", "")
            hof_cluster = analysis.get('largest_hydrophobic_cluster')
            hif_cluster = analysis.get('largest_hydrophilic_cluster')
            hof_positions = hof_cluster.get('positions', []) if hof_cluster else []
            hif_positions = hif_cluster.get('positions', []) if hif_cluster else []
            seq_plot = reconstruct_sequence_from_v5(analysis)
            metrics = compute_surface_metrics(seq_plot, analysis, hof_positions, hif_positions)
            pdb_feat = {
                "HoF_HMom_PDB": round(hof_hmom, 4) if isinstance(hof_hmom, (int, float)) else "",
                "HMom_PDB": round(overall_hmom, 4) if isinstance(overall_hmom, (int, float)) else "",
                "HiF_Cluster_Count": hif_count,
                "HiF_HiMom_PDB": round(hif_himom, 4) if isinstance(hif_himom, (int, float)) else "",
                "Center_Angle_deg": round(center_angle, 2) if isinstance(center_angle, (int, float)) else "",
                "HCS4_HoF_Mean_PDB": analysis.get("hcs4_mean", ""),
                "HCS4_HoF_Mean_SD_PDB": analysis.get("hcs4_sd", ""),
                "HCS4_HoF_Num_PDB": analysis.get("hcs4_num", 0),
                "HCS3_HoF_Mean_PDB": analysis.get("hcs3_mean", ""),
                "HCS3_HoF_Mean_SD_PDB": analysis.get("hcs3_sd", ""),
                "HCS3_HoF_Num_PDB": analysis.get("hcs3_num", 0),
                "HCS_Pair_Num_PDB": analysis.get("hcs_pair_num", 0),
                "HoF_HiF_Distance": round(metrics["hof_hif_distance"], 2) if metrics["hof_hif_distance"] else "",
                "HoF_Width": round(metrics["hof_width"], 2) if metrics["hof_width"] else "",
                "HoF_Height": round(metrics["hof_height"], 2) if metrics["hof_height"] else "",
                "HoF_Area": round(metrics["hof_area"], 2) if metrics["hof_area"] else "",
                "HiF_Width": round(metrics["hif_width"], 2) if metrics["hif_width"] else "",
                "HiF_Height": round(metrics["hif_height"], 2) if metrics["hif_height"] else "",
                "HiF_Area": round(metrics["hif_area"], 2) if metrics["hif_area"] else "",
                "Delta_HoF_HiF_X": round(metrics["delta_x"], 2) if metrics["delta_x"] else "",
                "Delta_HoF_HiF_Y": round(metrics["delta_y"], 2) if metrics["delta_y"] else "",
                "_hcs_pair_values": analysis.get("hcs_pair_values", []),
                "_hcs_pair_compositions": analysis.get("hcs_pair_compositions", []),
                "_analysis": analysis
            }
            if generate_plots and _MATPLOTLIB_AVAILABLE:
                safe_name = re.sub(r'[^\w\-.]', '_', name)
                subdirs = ['HoF','HiF','HoF_PDB','HiF_PDB','HoF_PDB_Mark','HiF_PDB_Mark','HoF_HiF_PDB','HoF_HiF_PDB_Mark']
                for sd in subdirs:
                    os.makedirs(os.path.join(output_dir, sd), exist_ok=True)
                draw_helix_wheel_hof(seq_plot, os.path.join(output_dir,'HoF',f"{safe_name}_HoF.png"), analysis)
                draw_helix_wheel_hif(seq_plot, os.path.join(output_dir,'HiF',f"{safe_name}_HiF.png"), analysis)
                draw_helix_wheel_hof(seq_plot, os.path.join(output_dir,'HoF_PDB',f"{safe_name}_HoF_PDB.png"), analysis)
                draw_helix_wheel_hif(seq_plot, os.path.join(output_dir,'HiF_PDB',f"{safe_name}_HiF_PDB.png"), analysis)
                draw_helix_wheel_hof_hif(seq_plot, os.path.join(output_dir,'HoF_HiF_PDB',f"{safe_name}_HoF_HiF_PDB.png"), analysis)
                draw_helix_wheel_hof_with_mark(seq_plot, os.path.join(output_dir,'HoF_PDB_Mark',f"{safe_name}_HoF_PDB_Mark.png"), analysis, hof_cluster)
                draw_helix_wheel_hif_with_mark(seq_plot, os.path.join(output_dir,'HiF_PDB_Mark',f"{safe_name}_HiF_PDB_Mark.png"), analysis, hif_cluster)
                draw_helix_wheel_hof_hif_mark(seq_plot, os.path.join(output_dir,'HoF_HiF_PDB_Mark',f"{safe_name}_HoF_HiF_PDB_Mark.png"),
                                              analysis, hof_positions, hif_positions)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            pdb_feat = {
                "HoF_HMom_PDB": "", "HMom_PDB": "", "HiF_Cluster_Count": 0, "HiF_HiMom_PDB": "",
                "Center_Angle_deg": "", "HCS4_HoF_Mean_PDB": "", "HCS4_HoF_Mean_SD_PDB": "",
                "HCS4_HoF_Num_PDB": 0, "HCS3_HoF_Mean_PDB": "", "HCS3_HoF_Mean_SD_PDB": "",
                "HCS3_HoF_Num_PDB": 0, "HCS_Pair_Num_PDB": 0,
                "HoF_HiF_Distance": "",
                "HoF_Width": "", "HoF_Height": "", "HoF_Area": "",
                "HiF_Width": "", "HiF_Height": "", "HiF_Area": "",
                "Delta_HoF_HiF_X": "", "Delta_HoF_HiF_Y": "",
                "_hcs_pair_values": [], "_hcs_pair_compositions": [], "_analysis": None
            }
    row_dict = {
        "Name": name,
        "Sequences": seq,
        "Exp_Log2_MIC": exp,
        **basic,
        **{k: v for k, v in pdb_feat.items() if not k.startswith("_")}
    }
    row_dict['_hcs_pair_values'] = pdb_feat.get('_hcs_pair_values', [])
    row_dict['_hcs_pair_compositions'] = pdb_feat.get('_hcs_pair_compositions', [])
    row_dict['_analysis'] = pdb_feat.get('_analysis')
    return row_dict

def generate_full_summary_excel(rows_data, out_dir, base_name):
    """生成完整摘要 Excel，包含动态 HCS_Pair 列"""
    results = []
    max_pair = 0
    for row in rows_data:
        name = row["Name"]
        seq = row["Sequences"]
        exp = row["Exp_Log2_MIC"]
        json_path = os.path.join(out_dir, f"{name}_sidechain.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
        else:
            data = {
                "overall_hmom_actual": row.get("HMom_PDB", ""),
                "largest_hydrophobic_cluster": {"composition": row.get("HoF_HMom_PDB_Num", ""),
                                                "hmom_actual": row.get("HoF_HMom_PDB", "")},
                "hydrophilic_clusters_sorted": [{"composition": row.get("HiF_Cluster_Num", ""),
                                                "hydrophilic_moment_actual": row.get("HiF_HiMom_PDB", "")}] if row.get("HiF_Cluster_Count",0)>0 else [],
                "center_angle_deg": row.get("Center_Angle_deg", ""),
                "hcs4_mean": row.get("HCS4_HoF_Mean_PDB", ""),
                "hcs4_sd": row.get("HCS4_HoF_Mean_SD_PDB", ""),
                "hcs4_num": row.get("HCS4_HoF_Num_PDB", 0),
                "hcs3_mean": row.get("HCS3_HoF_Mean_PDB", ""),
                "hcs3_sd": row.get("HCS3_HoF_Mean_SD_PDB", ""),
                "hcs3_num": row.get("HCS3_HoF_Num_PDB", 0),
                "hcs_pair_num": row.get("HCS_Pair_Num_PDB", 0),
                "hcs_pair_values": row.get("_hcs_pair_values", []),
                "hcs_pair_compositions": row.get("_hcs_pair_compositions", []),
                "hof_hif_distance": row.get("HoF_HiF_Distance", ""),
                "hof_width": row.get("HoF_Width", ""),
                "hof_height": row.get("HoF_Height", ""),
                "hof_area": row.get("HoF_Area", ""),
                "hif_width": row.get("HiF_Width", ""),
                "hif_height": row.get("HiF_Height", ""),
                "hif_area": row.get("HiF_Area", ""),
                "delta_x": row.get("Delta_HoF_HiF_X", ""),
                "delta_y": row.get("Delta_HoF_HiF_Y", ""),
            }
        full_composition = ",".join(f"{aa}{i+1}" for i, aa in enumerate(seq))
        hmom_theoretical_full = calculate_full_theoretical_moment(seq)
        valid_res = [f"{info['one_letter']}{info['index']}" for idx, info in data["residues"].items()
                     if "angle_deg" in info] if "residues" in data else []
        hmom_pdb_composition = ",".join(valid_res)
        hmom_pdb = data.get("overall_hmom_actual", "")
        lhc = data.get("largest_hydrophobic_cluster")
        if lhc:
            hof_num = lhc.get("composition", "")
            hof_hmom_pdb = lhc.get("hmom_actual", "")
        else:
            hof_num = ""
            hof_hmom_pdb = ""
        hif_list = data.get("hydrophilic_clusters_sorted", [])
        hif_count = len(hif_list)
        if hif_list:
            hif_num = hif_list[0].get("composition", "")
            hif_himom = hif_list[0].get("hydrophilic_moment_actual", "")
        else:
            hif_num = ""
            hif_himom = ""
        center_angle = data.get("center_angle_deg", "")
        hcs4_mean = data.get("hcs4_mean", "")
        hcs4_sd = data.get("hcs4_sd", "")
        hcs4_num = data.get("hcs4_num", 0)
        hcs3_mean = data.get("hcs3_mean", "")
        hcs3_sd = data.get("hcs3_sd", "")
        hcs3_num = data.get("hcs3_num", 0)
        hcs_pair_num = data.get("hcs_pair_num", 0)
        hcs_pair_vals = data.get("hcs_pair_values", [])
        hcs_pair_comps = data.get("hcs_pair_compositions", [])
        if hcs_pair_vals:
            pairs_mean = round(np.mean(hcs_pair_vals), 4)
            pairs_sd = round(np.std(hcs_pair_vals, ddof=1), 4) if len(hcs_pair_vals) > 1 else 0.0
        else:
            pairs_mean = 0.0
            pairs_sd = 0.0
        max_pair = max(max_pair, len(hcs_pair_vals))

        row_dict = {
            "Name": name,
            "Sequences": seq,
            "Exp_Log2_MIC": exp,
            "HoF_Num": hof_num,
            "HoF_HMom_PDB_Num": hof_num,
            "HoF_HMom_PDB": hof_hmom_pdb,
            "HMom_Num": full_composition,
            "HMom": hmom_theoretical_full,
            "HMom_PDB_Num": hmom_pdb_composition,
            "HMom_PDB": hmom_pdb,
            "HiF_Cluster_Count": hif_count,
            "HiF_Cluster_Num": hif_num,
            "HiF_HiMom_PDB_Num": hif_num,
            "HiF_HiMom_PDB": hif_himom,
            "Center_Angle_deg": center_angle if center_angle is not None else "",
            "HCS4_HoF_Mean_PDB": hcs4_mean,
            "HCS4_HoF_Mean_SD_PDB": hcs4_sd,
            "HCS4_HoF_Num_PDB": hcs4_num,
            "HCS3_HoF_Mean_PDB": hcs3_mean,
            "HCS3_HoF_Mean_SD_PDB": hcs3_sd,
            "HCS3_HoF_Num_PDB": hcs3_num,
            "HCS_Pair_Num_PDB": hcs_pair_num,
            "HCS_Pairs_Mean_PDB": pairs_mean,
            "HCS_Pairs_Mean_SD_PDB": pairs_sd,
            "HoF_HiF_Distance": data.get("hof_hif_distance", ""),
            "HoF_Width": data.get("hof_width", ""),
            "HoF_Height": data.get("hof_height", ""),
            "HoF_Area": data.get("hof_area", ""),
            "HiF_Width": data.get("hif_width", ""),
            "HiF_Height": data.get("hif_height", ""),
            "HiF_Area": data.get("hif_area", ""),
            "Delta_HoF_HiF_X": data.get("delta_x", ""),
            "Delta_HoF_HiF_Y": data.get("delta_y", ""),
        }
        for i, (comp, val) in enumerate(zip(hcs_pair_comps, hcs_pair_vals), start=1):
            row_dict[f"HCS_Pair{i}_PDB_Num"] = comp
            row_dict[f"HCS_Pair{i}_PDB"] = val
        results.append(row_dict)

    df_full = pd.DataFrame(results)
    for i in range(1, max_pair + 1):
        col_num = f"HCS_Pair{i}_PDB_Num"
        col_val = f"HCS_Pair{i}_PDB"
        if col_num not in df_full.columns:
            df_full[col_num] = ""
        if col_val not in df_full.columns:
            df_full[col_val] = 0.0
    fixed_cols = [
        "Name", "Sequences", "Exp_Log2_MIC",
        "HoF_Num", "HoF_HMom_PDB_Num", "HoF_HMom_PDB",
        "HMom_Num", "HMom", "HMom_PDB_Num", "HMom_PDB",
        "HiF_Cluster_Count", "HiF_Cluster_Num", "HiF_HiMom_PDB_Num", "HiF_HiMom_PDB",
        "Center_Angle_deg",
        "HCS4_HoF_Mean_PDB", "HCS4_HoF_Mean_SD_PDB", "HCS4_HoF_Num_PDB",
        "HCS3_HoF_Mean_PDB", "HCS3_HoF_Mean_SD_PDB", "HCS3_HoF_Num_PDB",
        "HCS_Pair_Num_PDB", "HCS_Pairs_Mean_PDB", "HCS_Pairs_Mean_SD_PDB",
        "HoF_HiF_Distance",
        "HoF_Width", "HoF_Height", "HoF_Area",
        "HiF_Width", "HiF_Height", "HiF_Area",
        "Delta_HoF_HiF_X", "Delta_HoF_HiF_Y"
    ]
    pair_cols = []
    for i in range(1, max_pair + 1):
        pair_cols.append(f"HCS_Pair{i}_PDB_Num")
        pair_cols.append(f"HCS_Pair{i}_PDB")
    full_cols = fixed_cols + pair_cols
    full_cols = [c for c in full_cols if c in df_full.columns]
    df_full = df_full[full_cols]
    excel_path = os.path.join(out_dir, f"{base_name}_HELNET_Summary.xlsx")
    df_full.to_excel(excel_path, index=False)
    print(f"Full summary Excel saved: {excel_path}")

def generate_model_data(seq_tsv, pdb_dir, output_dir, feature_model_txt=None,
                        sheet_name="Model_Data", clean=False, generate_txt=False,
                        generate_plots=False, workers=None):
    if clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    df_seq = pd.read_csv(seq_tsv, sep='\t')
    required_cols = ['Name', 'Sequences', 'Exp_Log2_MIC']
    for col in required_cols:
        if col not in df_seq.columns:
            raise ValueError(f"Sequence file must contain column: {col}")

    tasks = []
    for _, row in df_seq.iterrows():
        tasks.append((
            row["Name"], row["Sequences"], row["Exp_Log2_MIC"],
            pdb_dir, output_dir, generate_txt, generate_plots
        ))

    if workers is None:
        workers = max(1, cpu_count() - 1)
    print(f"Using {workers} worker processes for parallel processing.")

    with Pool(processes=workers) as pool:
        results = pool.starmap(process_single_pdb, tasks)

    model_rows = []
    for r in results:
        clean_row = {k: v for k, v in r.items() if not k.startswith('_')}
        model_rows.append(clean_row)
    df_model = pd.DataFrame(model_rows)

    if feature_model_txt and os.path.exists(feature_model_txt):
        with open(feature_model_txt, "r") as f:
            col_order = [line.strip() for line in f if line.strip()]
    else:
        col_order = [
            "Name", "Sequences", "Exp_Log2_MIC",
            "Length", "Hyd", "HMom", "z", "FreqPolar", "FreqNonPolar",
            "HoF_HMom_PDB", "HMom_PDB", "HiF_Cluster_Count", "HiF_HiMom_PDB",
            "Center_Angle_deg",
            "HCS4_HoF_Mean_PDB", "HCS4_HoF_Mean_SD_PDB", "HCS4_HoF_Num_PDB",
            "HCS3_HoF_Mean_PDB", "HCS3_HoF_Mean_SD_PDB", "HCS3_HoF_Num_PDB",
            "HCS_Pair_Num_PDB", "HoF_HiF_Distance",
            "HoF_Width", "HoF_Height", "HoF_Area",
            "HiF_Width", "HiF_Height", "HiF_Area",
            "Delta_HoF_HiF_X", "Delta_HoF_HiF_Y"
        ]
    existing = [c for c in col_order if c in df_model.columns]
    extra = [c for c in df_model.columns if c not in existing]
    df_model = df_model[existing + extra]
    model_excel = os.path.join(output_dir, "Model_Data.xlsx")
    df_model.to_excel(model_excel, index=False, sheet_name=sheet_name)
    print(f"Model_Data.xlsx saved: {model_excel}")

    base_name = os.path.splitext(os.path.basename(seq_tsv))[0]
    generate_full_summary_excel(results, output_dir, base_name)

def main():
    parser = argparse.ArgumentParser(description="HELNET_PDB_Integrated - Full pipeline with multiprocessing, plotting, and full summary")
    parser.add_argument("--seq_file", required=True, help="Sequence TSV (Name, Sequences, Exp_Log2_MIC)")
    parser.add_argument("--pdb_dir", required=True, help="Directory with PDB files (named as Name.pdb)")
    parser.add_argument("--output_dir", default="./helnet_output", help="Output directory")
    parser.add_argument("--feature_model", help="Feature_Model.txt defining columns for Model_Data.xlsx")
    parser.add_argument("--sheet_name", default="Model_Data", help="Sheet name for Model_Data.xlsx")
    parser.add_argument("--clean", action="store_true", help="Remove output_dir before processing")
    parser.add_argument("--txt", action="store_true", help="Generate TXT reports for each PDB")
    parser.add_argument("--HELNET", action="store_true", help="Generate helix wheel plots")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 1)")
    args = parser.parse_args()

    generate_model_data(
        seq_tsv=args.seq_file,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        feature_model_txt=args.feature_model,
        sheet_name=args.sheet_name,
        clean=args.clean,
        generate_txt=args.txt,
        generate_plots=args.HELNET,
        workers=args.workers,
    )

if __name__ == "__main__":
    main()