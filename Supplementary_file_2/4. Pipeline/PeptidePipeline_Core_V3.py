# PeptidePipeline_Core_V3.py
# Original Step2 face rules + all Stage-1 features
# Enhanced: calls amp_face_features.py to add ALL AMP face-related features
# Output: tab-separated txt to stdout (use > out.txt)

import math
import sys
import numpy as np
from collections import Counter
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Any

# ============================ Constants (from Core.py) ============================

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

# ============================ Core Functions (from Core.py) ============================

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

# ============================ Multi-Round Helical Wheel Functions ============================

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
    """查找疏水面（保持原 Core_V2 规则）
    返回：
      - hydrophobic_face_wheel: List[int]  # 螺旋轮位置（0..round1_length-1），用于角度/离散度等计算
      - best_start: int
      - hydrophobic_face_seq: str          # 疏水面残基串（按当前多轮拼接规则）
      - hydrophobic_face_linear_idxs: List[int]  # 对应原始序列的线性下标（0-based），与 hydrophobic_face_seq 的拼接顺序一致
    """
    length = len(sequence)
    if length < ROUND_SIZE:
        return [], -1, "", []

    rounds = get_round_sequences(sequence)
    round1_seq = rounds[0]
    round2_seq = rounds[1]
    round3_seq = rounds[2]
    round1_length = len(round1_seq)

    # wheel 映射：wheel 位置 -> 原始线性 index（仅 round1）
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
                # wheel pos (用于角度/离散度/面宽等)
                current_face.append(pos)

                # round1：wheel pos -> 线性 index
                current_face_seq.append(aa1)
                current_face_linear_idxs.append(mapping[pos])

                # round2：线性 index = round1_length + pos
                if round2_pos_exists:
                    aa2 = round2_seq[pos]
                    if is_hydrophobic_amino_acid(aa2, 2):
                        current_face_seq.append(aa2)
                        idx2 = round1_length + pos
                        if 0 <= idx2 < length:
                            current_face_linear_idxs.append(idx2)

                # round3：线性 index = round1_length + len(round2_seq) + pos
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

    # 验证亲水面是否都是亲水性氨基酸
    for pos in hydrophilic_face:
        aa = round1_wheel_seq[pos]
        if aa not in HYDROPHILIC_FACE_RESIDUES:
            return []

    return hydrophilic_face

def find_helical_faces_multiround(sequence: str) -> Tuple[str, str, List[int], List[int], List[int], List[int]]:
    """查找螺旋面（保持原 Core_V2 规则）

    返回：
      - hydrophobic_face_seq: str
      - hydrophilic_face_seq: str
      - hyd_positions_wheel: List[int]      # 螺旋轮位置（用于角度/离散度/amp_face_features）
      - phil_positions_wheel: List[int]     # 螺旋轮位置
      - hyd_positions_linear: List[int]     # 线性 index（0-based），与 hydrophobic_face_seq 拼接顺序一致（含 round2/round3）
      - phil_positions_linear: List[int]    # 线性 index（0-based，仅 round1）
    """
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


# ============================ Face Dispersion Calculation ============================

def calculate_hydrophobic_face_dispersion(face_positions: List[int], sequence_length: int) -> float:
    """计算疏水面在螺旋轮上的角度离散度（0~1）"""
    if not face_positions:
        return 0.0
    angles = [math.radians(pos * DEGREES_PER_RESIDUE) for pos in face_positions]
    sum_cos = sum(math.cos(a) for a in angles)
    sum_sin = sum(math.sin(a) for a in angles)
    R = math.sqrt(sum_cos**2 + sum_sin**2) / len(angles)
    return 1.0 - R

# ============================ Weighted HoF Functions ============================

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

# ============================ Face Frequency Features ============================

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

# ============================ Original V2 Functions ============================

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

# ============================ AMP feature integration (from amp_face_features.py) ============================

def _compute_amp_features_safe(seq: str, hyd_positions: List[int], phil_positions: List[int]) -> Dict[str, float]:
    """
    调用 amp_face_features.py，输出 AMP_* 列（若依赖缺失则全 0，不报错）
    """
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
        out = compute_amp_face_features(seq, hyd_positions, phil_positions)  # legacy keys
        # legacy -> AMP_* 映射（保证输出列名与 Step2_Analysis 一致）
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

# ============================ Main Feature Computation Function ============================

def compute_features(seq: str, name: str) -> Dict[str, Any]:
    """计算所有特征的函数"""
    L = len(seq)

    # -------- empty row (still returns full header set) --------
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

        # faces (core fields)
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
            # 添加缺失的列
            "HydrophobicFaceSeqIdx": "",
            "HydrophilicFaceSeqIdx": "",
            "HydrophobicFaceSeq": "None",
            "HydrophilicFaceSeq": "None",
        })

        empty.update(_compute_amp_features_safe(seq, [], []))
        return empty

    # ==================== basic ====================
    hyd_mean = calculate_mean_hydrophobicity(seq)
    hmom = calculate_mean_amphipathic_moment(seq)
    z = calculate_net_charge(seq)

    aa_counts = calculate_aa_counts(seq)
    aa_freqs = {aa: (aa_counts[aa] / L if L > 0 else 0.0) for aa in AA_ORDER}

    freq_polar = calculate_polar_residues_ratio(seq)
    freq_non_polar = calculate_non_polar_residues_ratio(seq)

    # ==================== faces (KEEP Core_V2 RULES) ====================
    hydrophobic_face_seq, hydrophilic_face_seq, hyd_positions, phil_positions, hyd_linear_idxs, phil_linear_idxs = find_helical_faces_multiround(seq)
    
    # 添加：生成疏水面索引字符串
    hydrophobic_face_seq_idx_str = ",".join(map(str, hyd_linear_idxs)) if hyd_linear_idxs else ""
    hydrophilic_face_seq_idx_str = ",".join(map(str, phil_linear_idxs)) if phil_linear_idxs else ""

    # freq features on faces
    face_freq_features = calculate_all_face_frequency_features(hydrophobic_face_seq, hydrophilic_face_seq)

    hyd_hydrophobic_face = calculate_mean_hydrophobicity(hydrophobic_face_seq) if hydrophobic_face_seq != "None" else 0.0
    h_mom_hydrophobic_face = calculate_mean_amphipathic_moment(hydrophobic_face_seq) if hydrophobic_face_seq != "None" else 0.0
    hyd_hydrophilic_face = calculate_mean_hydrophobicity(hydrophilic_face_seq) if hydrophilic_face_seq != "None" else 0.0
    h_mom_hydrophilic_face = calculate_mean_amphipathic_moment(hydrophilic_face_seq) if hydrophilic_face_seq != "None" else 0.0

    hydrophobic_face_length = 0 if hydrophobic_face_seq == "None" else len(hydrophobic_face_seq)
    hydrophobic_face_dispersion = calculate_hydrophobic_face_dispersion(hyd_positions, L)

    # ==================== physchem ====================
    d_value = calculate_d_value(hmom, z)
    helix_type = determine_helix_type(d_value)
    pI = calculate_isoelectric_point(seq)
    charge_at_pH7 = calculate_charge_at_ph(seq, 7.0)

    # ==================== weighted HoF ====================
    weighted_hf_peptide = weighted_hf_score(seq, list(range(L)))
    weighted_hf_peptide = weighted_hf_score(seq, list(range(L)))
    # Use true helix-face linear indices (0-based) for weighted HoF on faces
    weighted_hyd_face_positions = list(hyd_linear_idxs or [])
    weighted_philic_face_positions = list(phil_linear_idxs or [])
    weighted_hf_hydrophobic_face = weighted_hf_score(seq, weighted_hyd_face_positions)
    weighted_hf_hydrophilic_face = weighted_hf_score(seq, weighted_philic_face_positions)

    # ==================== assemble ====================
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
        # 添加缺失的列
        "HydrophobicFaceSeqIdx": hydrophobic_face_seq_idx_str,
        "HydrophilicFaceSeqIdx": hydrophilic_face_seq_idx_str,
        "HydrophobicFaceSeq": hydrophobic_face_seq,
        "HydrophilicFaceSeq": hydrophilic_face_seq,
    })

    # ==================== AMP_* (from amp_face_features.py) ====================
        # AMP_* features expect 0-based LINEAR residue indices because angles use (i*100°) mod 360.
    # Our face-finding returns wheel positions (angle-sorted indices) plus linear indices; use the linear indices (round1 only) here.
    _amp_round1_len = min(18, len(seq))
    hyd_positions_amp = [i for i in hyd_linear_idxs if 0 <= i < _amp_round1_len]
    phil_positions_amp = [i for i in phil_linear_idxs if 0 <= i < _amp_round1_len]
    features.update(_compute_amp_features_safe(seq, hyd_positions_amp, phil_positions_amp))

    return features

# ============================ Generate Header ============================

def generate_header() -> str:
    """生成表头行"""
    sample_features = compute_features("AAAAAAAAAAAAAAAAAA", "Sample")  # length>=18 to ensure face keys exist
    header_fields = list(sample_features.keys())
    return "\t".join(header_fields)

# ============================ Main Execution ============================

if __name__ == "__main__":
    header_only = False
    input_source = None

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--header":
                header_only = True
            elif not arg.startswith("--"):
                input_source = arg

    if header_only:
        print(generate_header())
        sys.exit(0)

    print(generate_header())

    if input_source:
        try:
            with open(input_source, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    seq = line.strip()
                    if not seq:
                        continue
                    feats = compute_features(seq, f"Sequence_{i}")
                    print("\t".join(str(feats[k]) for k in feats))
        except FileNotFoundError:
            print(f"错误: 文件 '{input_source}' 未找到", file=sys.stderr)
            sys.exit(1)
    else:
        for i, line in enumerate(sys.stdin, 1):
            seq = line.strip()
            if not seq:
                continue
            feats = compute_features(seq, f"Sequence_{i}")
            print("\t".join(str(feats[k]) for k in feats))