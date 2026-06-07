"""
amp_features_basic_V2.py

V2: merge of basic + basic_V3 + AMP_* features
- keeps original basic features: amino-acid frequencies, Hyd, HMom, z, pI, FreqPolar/NonPolar
- keeps amphiphilic/face/AMP_* features
- adds V3 features:
  - PositiveFaceAngleDeg_CoreV3: 正电荷面角度（基于helical face positions）
  - PositiveFaceAngleDeg_Continuous: 正电荷面角度（基于整个序列）
  - HydrophobicFaceAngleDeg_CoreV3: 疏水面角度（基于helical face positions）
  - HydrophobicFaceAngleDeg_Continuous: 疏水面角度（基于整个序列）
  - Hydrophobic_vs_CoreV3_PosFaceAngleDeg: 疏水vs正电荷面角度差
  - Hydrophobic_vs_Continuous_PosFaceAngleDeg: 疏水vs正电荷面角度差
  - Positive_vs_HydrophobicFaceAngleDeg: 正电荷vs疏水面角度差
- fixes circular import issue by defining all functions internally
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

from PeptidePipeline_Core_V3 import (
    calculate_mean_hydrophobicity,
    calculate_mean_amphipathic_moment,
    calculate_net_charge,
    calculate_isoelectric_point,
    calculate_polar_residues_ratio,
    calculate_non_polar_residues_ratio,
    AA_ORDER,
    find_helical_faces_multiround,
)

DEGREES_PER_RESIDUE = 100.0  # alpha-helix wheel
FAUCHERE_PLISKA_HYDROPHOBICITY = {
    'A': 0.310, 'R': -1.010, 'N': -0.600, 'D': -0.770, 'C': 1.540,
    'Q': -0.220, 'E': -0.640, 'G': 0.000, 'H': 0.130, 'I': 1.800,
    'L': 1.700, 'K': -0.990, 'M': 1.230, 'F': 1.790, 'P': 0.720,
    'S': -0.040, 'T': 0.260, 'W': 2.250, 'Y': 0.960, 'V': 1.220
}


# =========================================================
# Helper functions (V3 angle features)
# =========================================================

def circular_distance_deg(a: float, b: float) -> float:
    """Minimal circular distance in degrees in [0,180]."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def compute_positive_face_angle(seq: str, positions: list) -> float:
    """
    Compute the minimal covering arc angle (degrees) for positive residues (K/R)
    within the given positions list on the helical wheel.
    """
    if not seq or not positions:
        return 0.0
    pos_indices = [i for i in positions if seq[i] in ('K', 'R')]
    if not pos_indices:
        return 0.0
    angles_deg = [(i * DEGREES_PER_RESIDUE) % 360.0 for i in pos_indices]
    angles_deg.sort()
    angles_dup = angles_deg + [a + 360.0 for a in angles_deg]
    best_arc = 360.0
    j = 0
    for i in range(len(angles_deg)):
        while j < i + len(angles_deg) and (angles_dup[j] - angles_dup[i]) < 360.0:
            j += 1
        arc_width = angles_dup[j-1] - angles_dup[i]
        if arc_width < best_arc:
            best_arc = arc_width
    return best_arc % 360.0


def compute_hydrophobic_face_angle(seq: str, positions: list) -> float:
    """
    Compute the minimal covering arc angle (degrees) for hydrophobic residues
    within the given positions list on the helical wheel.
    A residue is considered hydrophobic if its Fauchere-Pliska hydrophobicity > 0.
    """
    if not seq or not positions:
        return 0.0
    hyd_indices = [i for i in positions if FAUCHERE_PLISKA_HYDROPHOBICITY.get(seq[i], 0.0) > 0]
    if not hyd_indices:
        return 0.0
    angles_deg = [(i * DEGREES_PER_RESIDUE) % 360.0 for i in hyd_indices]
    angles_deg.sort()
    angles_dup = angles_deg + [a + 360.0 for a in angles_deg]
    best_arc = 360.0
    j = 0
    for i in range(len(angles_deg)):
        while j < i + len(angles_deg) and (angles_dup[j] - angles_dup[i]) < 360.0:
            j += 1
        arc_width = angles_dup[j-1] - angles_dup[i]
        if arc_width < best_arc:
            best_arc = arc_width
    return best_arc % 360.0


# =========================================================
# AMP features core functions
# =========================================================

def _hcs3_hf(seq: str, hyd_set: set, hs: list) -> float:
    """Face-restricted hydrophobic coupling score at offset 3:
    HCS3_HoF = mean(h_i*h_{i+3}) over pairs (i,i+3) where both indices are in hyd_set."""
    if not seq or not hyd_set:
        return 0.0
    L = len(seq)
    pairs = 0
    acc = 0.0
    for i in hyd_set:
        j = i + 3
        if 0 <= i < L and j < L and j in hyd_set:
            acc += float(hs[i]) * float(hs[j])
            pairs += 1
    return acc / pairs if pairs > 0 else 0.0


def _pair_scores_hf(seq: str, hyd_set: set, hs: list, k: int) -> list:
    """Ordered per-pair scores h_i*h_{i+k} for (i,i+k) within hyd_set.
    Ordering is ascending i for determinism."""
    if not seq or not hyd_set:
        return []
    L = len(seq)
    out = []
    for i in sorted(hyd_set):
        j = i + k
        if 0 <= i < L and j < L and j in hyd_set:
            out.append(float(hs[i]) * float(hs[j]))
    return out


def _hcs4_hf(seq: str, hyd_set: set, hs: list) -> float:
    """
    Face-restricted hydrophobic coupling score at offset 4:
      HCS4_HoF = mean( h_i * h_{i+4} ) over pairs (i, i+4) where BOTH indices are in hyd_set.
    """
    if not seq or not hyd_set:
        return 0.0
    L = len(seq)
    pairs = 0
    acc = 0.0
    for i in hyd_set:
        j = i + 4
        if 0 <= i < L and j < L and j in hyd_set:
            acc += float(hs[i]) * float(hs[j])
            pairs += 1
    return acc / pairs if pairs > 0 else 0.0


def _angle_rad(i: int) -> float:
    return math.radians((i * DEGREES_PER_RESIDUE) % 360.0)


def _circular_mean_vector(weights: Sequence[float], angles: Sequence[float]) -> float:
    """Returns magnitude of resultant vector."""
    sx = sum(w * math.cos(a) for w, a in zip(weights, angles))
    sy = sum(w * math.sin(a) for w, a in zip(weights, angles))
    return math.hypot(sx, sy)


def _weighted_circular_mean_deg(angle_deg: Sequence[float], weight: Sequence[float]) -> float:
    """Weighted circular mean of angles in degrees. Returns angle in [0,360)."""
    if not angle_deg:
        return 0.0
    sx = 0.0
    sy = 0.0
    for a_deg, w in zip(angle_deg, weight):
        a = math.radians(a_deg % 360.0)
        sx += w * math.cos(a)
        sy += w * math.sin(a)
    if abs(sx) < 1e-12 and abs(sy) < 1e-12:
        return 0.0
    ang = math.degrees(math.atan2(sy, sx)) % 360.0
    return ang


def _circular_distance_deg(a: float, b: float) -> float:
    """Minimal circular distance in degrees in [0,180]."""
    d = (a - b) % 360.0
    d = abs(d)
    return min(d, 360.0 - d)


def _min_covering_arc_deg(angle_deg: List[float], weight: List[float], target_frac: float = 0.80) -> float:
    """
    Minimal arc width (deg) covering >= target_frac of total weight on circle.
    Uses duplication trick on sorted angles.
    """
    if not angle_deg:
        return 0.0
    items = sorted(zip(angle_deg, weight), key=lambda x: x[0])
    ang = [a for a, _ in items]
    w = [max(0.0, float(x)) for _, x in items]
    total = sum(w)
    if total <= 0:
        return 0.0

    ang2 = ang + [a + 360.0 for a in ang]
    w2 = w + w

    best = 360.0
    j = 0
    acc = 0.0
    for i in range(len(ang)):
        while j < i + len(ang) and acc < target_frac * total:
            acc += w2[j]
            j += 1
        if acc >= target_frac * total:
            width = ang2[j - 1] - ang2[i]
            if width < best:
                best = width
        acc -= w2[i]
    return float(max(0.0, min(best, 360.0)))


def _charge_simple(aa: str) -> float:
    """Simple residue charge proxy at neutral pH."""
    if aa in ("K", "R"):
        return 1.0
    if aa in ("D", "E"):
        return -1.0
    if aa == "H":
        return 0.1
    return 0.0


def _dh_potential(qi: float, qj: float, rij: float, kappa: float = 1.0) -> float:
    """Debye-Huckel screened Coulomb (dimensionless proxy)."""
    if rij <= 1e-9:
        return 0.0
    return (qi * qj) * math.exp(-kappa * rij) / rij


def _helical_distance(i: int, j: int) -> float:
    """
    Crude alpha-helix residue-residue distance proxy (A) using:
    - axial rise 1.5 A per residue
    - radius ~2.3 A; angular separation 100 deg
    """
    dz = 1.5 * (j - i)
    ri = 2.3
    dtheta = _angle_rad(j) - _angle_rad(i)
    dr_xy = 2.0 * ri * math.sin(abs(dtheta) / 2.0)
    return math.hypot(dr_xy, dz)


def _positive_stripe_runs(seq: str, positions: List[int]) -> int:
    """
    Longest run of positive residues (K/R) along sequence order restricted to positions list.
    """
    pos_set = set(positions)
    longest = 0
    cur = 0
    for i, aa in enumerate(seq):
        if i not in pos_set:
            continue
        if aa in ("K", "R"):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def compute_amp_features(seq: str, hydrophobic_positions_0based: List[int], hydrophilic_positions_0based: List[int]) -> Dict[str, float]:
    """
    Returns dict with AMP_* keys.
    """
    L = len(seq)
    if L <= 0:
        return {k: 0.0 for k in [
            "AMP_muH_FP",
            "AMP_HydrophobicSectorWidthDeg",
            "AMP_HydrophilicSectorWidthDeg",
            "AMP_HydrophobicFaceWidthDeg",
            "AMP_HydrophilicFaceWidthDeg",
            "AMP_HydrophobicFaceCenterDeg",
            "AMP_HydrophilicFaceCenterDeg",
            "AMP_FaceCenterSeparationDeg",
            "AMP_FaceOppositionScore",
            "AMP_PoreTopologyScore",
            "AMP_HydrophobicContinuity",
            "AMP_muQ_simple",
            "AMP_phi_hydrophilic_DH",
            "AMP_phi_hydrophobic_DH",
            "AMP_AxialDipole_pz",
            "AMP_PosStripeContinuity",
            "AMP_PosStripeLongestRun",
            "AMP_AromaticAnchorScore",
            "AMP_HCS3_HoF",
            "AMP_HCS4_HoF",
            "AMP_Pairs1_HCS3_Score",
            "AMP_Pairs1_HCS4_Score",
            "AMP_Pairs2_HCS3_Score",
            "AMP_Pairs2_HCS4_Score",
            "delta_HCS4",
            "delta_HCS3",
            "Delta_HCS4",
            "Delta_HCS3",
        ]}

    # muH (hydrophobic moment magnitude, normalized by sum|h|)
    hs = [FAUCHERE_PLISKA_HYDROPHOBICITY.get(a, 0.0) for a in seq]
    ang = [_angle_rad(i) for i in range(L)]
    sx = sum(h * math.cos(a) for h, a in zip(hs, ang))
    sy = sum(h * math.sin(a) for h, a in zip(hs, ang))
    denom = sum(abs(h) for h in hs) or 1.0
    muH = math.hypot(sx, sy) / denom

    hyd_set = set(hydrophobic_positions_0based or [])
    phil_set = set(hydrophilic_positions_0based or [])

    AMP_HCS4_HoF = _hcs4_hf(seq, hyd_set, hs)
    AMP_HCS3_HoF = _hcs3_hf(seq, hyd_set, hs)

    # Per-pair coupling scores
    _pairs3_scores = _pair_scores_hf(seq, hyd_set, hs, k=3)
    _pairs4_scores = _pair_scores_hf(seq, hyd_set, hs, k=4)
    AMP_Pairs1_HCS3_Score = float(_pairs3_scores[0]) if len(_pairs3_scores) >= 1 else 0.0
    AMP_Pairs2_HCS3_Score = float(_pairs3_scores[1]) if len(_pairs3_scores) >= 2 else 0.0
    AMP_Pairs1_HCS4_Score = float(_pairs4_scores[0]) if len(_pairs4_scores) >= 1 else 0.0
    AMP_Pairs2_HCS4_Score = float(_pairs4_scores[1]) if len(_pairs4_scores) >= 2 else 0.0

    delta_HCS4 = AMP_Pairs1_HCS4_Score - AMP_Pairs2_HCS4_Score
    delta_HCS3 = AMP_Pairs1_HCS3_Score - AMP_Pairs2_HCS3_Score

    # Hydrophobic weights
    hyd_w_face: List[float] = []
    hyd_ang_face_deg: List[float] = []
    for i in sorted(hyd_set):
        if 0 <= i < L:
            h = hs[i]
            if h > 0:
                hyd_w_face.append(h)
                hyd_ang_face_deg.append((i * DEGREES_PER_RESIDUE) % 360.0)

    hyd_w: List[float] = list(hyd_w_face)
    hyd_ang_deg: List[float] = list(hyd_ang_face_deg)
    if not hyd_ang_deg:
        for i, h in enumerate(hs):
            if h > 0:
                hyd_w.append(h)
                hyd_ang_deg.append((i * DEGREES_PER_RESIDUE) % 360.0)

    AMP_HydrophobicFaceWidthDeg = _min_covering_arc_deg(hyd_ang_face_deg, hyd_w_face, target_frac=0.80) if hyd_ang_face_deg else 0.0
    AMP_HydrophobicSectorWidthDeg = _min_covering_arc_deg(hyd_ang_deg, hyd_w, target_frac=0.80)

    # Hydrophilic weights
    phil_w_face: List[float] = []
    phil_ang_face_deg: List[float] = []
    for i in sorted(phil_set):
        if 0 <= i < L:
            h = hs[i]
            w = (-h) if (h < 0) else 1.0
            phil_w_face.append(float(w))
            phil_ang_face_deg.append((i * DEGREES_PER_RESIDUE) % 360.0)

    phil_w: List[float] = list(phil_w_face)
    phil_ang_deg: List[float] = list(phil_ang_face_deg)
    if not phil_ang_deg:
        for i, h in enumerate(hs):
            if h < 0:
                phil_w.append(float(-h))
                phil_ang_deg.append((i * DEGREES_PER_RESIDUE) % 360.0)

    AMP_HydrophilicFaceWidthDeg = _min_covering_arc_deg(phil_ang_face_deg, phil_w_face, target_frac=0.80) if phil_ang_face_deg else 0.0
    AMP_HydrophilicSectorWidthDeg = _min_covering_arc_deg(phil_ang_deg, phil_w, target_frac=0.80)

    # Face centers and opposition
    hyd_center_deg = _weighted_circular_mean_deg(hyd_ang_face_deg, hyd_w_face) if hyd_ang_face_deg else (_weighted_circular_mean_deg(hyd_ang_deg, hyd_w) if hyd_ang_deg else 0.0)
    phil_center_deg = _weighted_circular_mean_deg(phil_ang_face_deg, phil_w_face) if phil_ang_face_deg else (_weighted_circular_mean_deg(phil_ang_deg, phil_w) if phil_ang_deg else 0.0)

    center_sep_deg = _circular_distance_deg(hyd_center_deg, phil_center_deg) if (hyd_center_deg or phil_center_deg) else 0.0
    face_opposition_score = (center_sep_deg / 180.0) if (hyd_center_deg or phil_center_deg) else 0.0

    # Hydrophobic continuity
    hyd_set = set(hydrophobic_positions_0based or [])
    cont = 0.0
    total_pair_w = 0.0
    for i in hyd_set:
        hi = FAUCHERE_PLISKA_HYDROPHOBICITY.get(seq[i], 0.0)
        for d, wcoef in ((3, 0.6), (4, 1.0), (7, 0.4)):
            j = i + d
            if j in hyd_set and j < L:
                hj = FAUCHERE_PLISKA_HYDROPHOBICITY.get(seq[j], 0.0)
                pair_w = abs(hi) * abs(hj)
                total_pair_w += pair_w
                cont += wcoef * pair_w
    hyd_continuity = cont / (total_pair_w or 1.0)

    # muQ (charge moment)
    qs = [_charge_simple(a) for a in seq]
    qx = sum(q * math.cos(a) for q, a in zip(qs, ang))
    qy = sum(q * math.sin(a) for q, a in zip(qs, ang))
    qden = sum(abs(q) for q in qs) or 1.0
    muQ = math.hypot(qx, qy) / qden

    # DH potential proxy
    def avg_face_phi(pos_set):
        charged = [i for i in pos_set if abs(qs[i]) > 1e-9]
        if len(charged) < 2:
            return 0.0
        acc = 0.0
        cnt = 0
        for a_i in range(len(charged)):
            i = charged[a_i]
            for a_j in range(a_i + 1, len(charged)):
                j = charged[a_j]
                r = _helical_distance(i, j)
                acc += _dh_potential(qs[i], qs[j], r, kappa=0.3)
                cnt += 1
        return acc / (cnt or 1)

    phi_phil = avg_face_phi(phil_set)
    phi_hyd = avg_face_phi(hyd_set)

    # Axial dipole proxy
    axial = sum(qs[i] * (1.5 * i) for i in range(L))

    # Positive stripe continuity
    pos_phil = [i for i in phil_set if seq[i] in ("K", "R")]
    pos_phil_set = set(pos_phil)
    cont2 = 0.0
    total2 = 0.0
    for i in pos_phil_set:
        for d, wcoef in ((3, 0.6), (4, 1.0), (7, 0.4)):
            j = i + d
            if j in pos_phil_set and j < L:
                total2 += 1.0
                cont2 += wcoef
    pos_stripe_cont = cont2 / (total2 or 1.0)
    pos_longest = float(_positive_stripe_runs(seq, list(phil_set)))

    # Aromatic anchor score
    center = float(hyd_center_deg)
    half = float(AMP_HydrophobicSectorWidthDeg) / 2.0
    boundary1 = (center - half) % 360.0
    boundary2 = (center + half) % 360.0

    def ang_dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    score = 0.0
    for i, aa in enumerate(seq):
        if aa not in ("W", "Y", "F"):
            continue
        a = (i * DEGREES_PER_RESIDUE) % 360.0
        if ang_dist(a, boundary1) <= 25.0 or ang_dist(a, boundary2) <= 25.0:
            score += 2.0 if aa == "W" else 1.0 if aa == "Y" else 0.8

    pore_topology = max(0.0, min(1.0, float(muH) * float(face_opposition_score) * float(pos_stripe_cont)))

    return {
        "AMP_muH_FP": float(muH),
        "AMP_HydrophobicSectorWidthDeg": float(AMP_HydrophobicSectorWidthDeg),
        "AMP_HydrophilicSectorWidthDeg": float(AMP_HydrophilicSectorWidthDeg),
        "AMP_HydrophobicFaceWidthDeg": float(AMP_HydrophobicFaceWidthDeg),
        "AMP_HydrophilicFaceWidthDeg": float(AMP_HydrophilicFaceWidthDeg),
        "AMP_HydrophobicFaceCenterDeg": float(hyd_center_deg),
        "AMP_HydrophilicFaceCenterDeg": float(phil_center_deg),
        "AMP_FaceCenterSeparationDeg": float(center_sep_deg),
        "AMP_FaceOppositionScore": float(face_opposition_score),
        "AMP_PoreTopologyScore": float(pore_topology),
        "AMP_HydrophobicContinuity": float(hyd_continuity),
        "AMP_muQ_simple": float(muQ),
        "AMP_phi_hydrophilic_DH": float(phi_phil),
        "AMP_phi_hydrophobic_DH": float(phi_hyd),
        "AMP_AxialDipole_pz": float(axial),
        "AMP_PosStripeContinuity": float(pos_stripe_cont),
        "AMP_PosStripeLongestRun": float(pos_longest),
        "AMP_AromaticAnchorScore": float(score),
        "AMP_HCS3_HoF": float(AMP_HCS3_HoF),
        "AMP_HCS4_HoF": float(AMP_HCS4_HoF),
        "AMP_Pairs1_HCS3_Score": float(AMP_Pairs1_HCS3_Score),
        "AMP_Pairs1_HCS4_Score": float(AMP_Pairs1_HCS4_Score),
        "AMP_Pairs2_HCS3_Score": float(AMP_Pairs2_HCS3_Score),
        "AMP_Pairs2_HCS4_Score": float(AMP_Pairs2_HCS4_Score),
        "delta_HCS4": float(delta_HCS4),
        "delta_HCS3": float(delta_HCS3),
        "Delta_HCS4": float(delta_HCS4),
        "Delta_HCS3": float(delta_HCS3),
    }


def compute_amp_face_features(
    seq: str,
    hydrophobic_positions_0based: List[int],
    hydrophilic_positions_0based: List[int],
) -> Dict[str, float]:
    """
    Returns AMP_* features with legacy aliases for backward compatibility.
    """
    feats = compute_amp_features(seq, hydrophobic_positions_0based, hydrophilic_positions_0based)
    if feats is None:
        feats = {}

    out: Dict[str, float] = dict(feats)

    # Add legacy (non-AMP_) aliases
    out.update({
        "muH_FP": feats.get("AMP_muH_FP", 0.0),
        "hydrophobic_face_width_deg": feats.get("AMP_HydrophobicFaceWidthDeg", 0.0),
        "hydrophilic_face_width_deg": feats.get("AMP_HydrophilicFaceWidthDeg", 0.0),
        "hydrophobic_face_center_deg": feats.get("AMP_HydrophobicFaceCenterDeg", 0.0),
        "hydrophilic_face_center_deg": feats.get("AMP_HydrophilicFaceCenterDeg", 0.0),
        "face_center_separation_deg": feats.get("AMP_FaceCenterSeparationDeg", 0.0),
        "face_opposition_score": feats.get("AMP_FaceOppositionScore", 0.0),
        "pore_topology_score": feats.get("AMP_PoreTopologyScore", 0.0),
        "hydrophobic_continuity": feats.get("AMP_HydrophobicContinuity", 0.0),
        "muQ_simple": feats.get("AMP_muQ_simple", 0.0),
        "phi_hydrophilic_DH": feats.get("AMP_phi_hydrophilic_DH", 0.0),
        "phi_hydrophobic_DH": feats.get("AMP_phi_hydrophobic_DH", 0.0),
        "axial_dipole_charge_pz": feats.get("AMP_AxialDipole_pz", 0.0),
        "positive_stripe_continuity": feats.get("AMP_PosStripeContinuity", 0.0),
        "positive_stripe_longest_run": feats.get("AMP_PosStripeLongestRun", 0.0),
        "aromatic_anchor_score": feats.get("AMP_AromaticAnchorScore", 0.0),
        "HCS3_HoF": feats.get("AMP_HCS3_HoF", 0.0),
        "HCS4_HoF": feats.get("AMP_HCS4_HoF", 0.0),
    })

    # Backward-compatible aliases
    out.setdefault("axial_dipole_pz", out["axial_dipole_charge_pz"])
    out.setdefault("pos_stripe_continuity", out["positive_stripe_continuity"])
    out.setdefault("pos_stripe_longest_run", out["positive_stripe_longest_run"])

    return out


# =========================================================
# Amphiphilic features
# =========================================================

def compute_amphiphilic_features(seq):
    """
    Compute amphiphilic features including face sequences and AMP_* features.
    Returns (feats_dict, hyd_lin_round1, phil_lin_round1).
    """
    (
        hyd_face_seq,
        phil_face_seq,
        hyd_wheel,
        phil_wheel,
        hyd_lin,
        phil_lin
    ) = find_helical_faces_multiround(seq)

    feats = {
        "HydrophobicFace": hyd_face_seq,
        "HydrophilicFace": phil_face_seq,
    }

    # Frequencies on faces
    def freq(face_seq, aa):
        return face_seq.count(aa) / len(face_seq) if face_seq and face_seq != "None" else 0.0

    for aa in "ALIVMPFWYG":
        feats[f"Freq{aa}_in_HydrophobicFace"] = freq(hyd_face_seq, aa)

    feats["FreqK_in_HydrophilicFace"] = freq(phil_face_seq, "K")
    feats["FreqR_in_HydrophilicFace"] = freq(phil_face_seq, "R")

    # Continuity
    def continuity(face_seq):
        if not face_seq or face_seq == "None":
            return 0.0
        max_run, cur = 1, 1
        for a, b in zip(face_seq[:-1], face_seq[1:]):
            cur = cur + 1 if a == b else 1
            max_run = max(max_run, cur)
        return max_run / len(face_seq)

    feats["FreqContinuity_in_HydrophobicFace"] = continuity(hyd_face_seq)
    feats["FreqContinuity_in_HydrophilicFace"] = continuity(phil_face_seq)

    # AMP_* features (round1 linear indices, 0-based)
    _amp_round1_len = min(18, len(seq))
    hyd_lin_round1 = [i for i in (hyd_lin or []) if 0 <= i < _amp_round1_len]
    phil_lin_round1 = [i for i in (phil_lin or []) if 0 <= i < _amp_round1_len]

    amp = compute_amp_face_features(seq, hyd_lin_round1, phil_lin_round1)
    if amp is None:
        amp = {}

    # Backfill AMP_* keys if legacy-only output
    if not any(k.startswith("AMP_") for k in amp.keys()):
        legacy_to_amp = {
            "muH_FP": "AMP_muH_FP",
            "hydrophobic_face_width_deg": "AMP_HydrophobicFaceWidthDeg",
            "hydrophilic_face_width_deg": "AMP_HydrophilicFaceWidthDeg",
            "face_center_separation_deg": "AMP_FaceCenterSeparationDeg",
            "face_opposition_score": "AMP_FaceOppositionScore",
            "hydrophobic_continuity": "AMP_HydrophobicContinuity",
            "positive_stripe_continuity": "AMP_PosStripeContinuity",
            "positive_stripe_longest_run": "AMP_PosStripeLongestRun",
            "axial_dipole_charge_pz": "AMP_AxialDipole_pz",
            "aromatic_anchor_score": "AMP_AromaticAnchorScore",
            "HCS3_HoF": "AMP_HCS3_HoF",
            "HCS4_HoF": "AMP_HCS4_HoF",
        }
        for lk, ak in legacy_to_amp.items():
            if lk in amp:
                amp[ak] = amp.get(lk, 0.0)

    feats.update(amp)
    return feats, hyd_lin_round1, phil_lin_round1


# =========================================================
# Main compute function
# =========================================================

def compute_basic_features(name: str, seq: str) -> dict:
    """
    Main entry point for computing all features.
    Returns dict with basic features + amphiphilic features + AMP_* features + V3 angle features.
    """
    L = len(seq)
    counter = Counter(seq)

    feats = {
        "Name": name,
        "Sequences": seq,
        "Length": L,
        "Hyd": calculate_mean_hydrophobicity(seq),
        "HMom": calculate_mean_amphipathic_moment(seq),
        "z": calculate_net_charge(seq),
        "pI": calculate_isoelectric_point(seq),
        "FreqPolar": calculate_polar_residues_ratio(seq),
        "FreqNonPolar": calculate_non_polar_residues_ratio(seq),
    }

    # Amino-acid frequencies (exact AA_ORDER from Core_V3)
    for aa in AA_ORDER:
        feats[f"Freq_{aa}"] = counter.get(aa, 0) / L if L else 0.0

    # Merge amphiphilic / face / AMP_* features
    amph_feats, hyd_lin, phil_lin = compute_amphiphilic_features(seq)
    feats.update(amph_feats)

    # ================== V3新增角度特征 ==================
    # PositiveFaceAngleDeg_CoreV3: 正电荷面角度（基于helical face positions）
    feats['PositiveFaceAngleDeg_CoreV3'] = compute_positive_face_angle(seq, phil_lin)

    # PositiveFaceAngleDeg_Continuous: 正电荷面角度（基于整个序列）
    feats['PositiveFaceAngleDeg_Continuous'] = compute_positive_face_angle(seq, list(range(L)))

    # HydrophobicFaceAngleDeg_CoreV3: 疏水面角度（基于helical face positions）
    feats['HydrophobicFaceAngleDeg_CoreV3'] = compute_hydrophobic_face_angle(seq, hyd_lin)

    # HydrophobicFaceAngleDeg_Continuous: 疏水面角度（基于整个序列）
    feats['HydrophobicFaceAngleDeg_Continuous'] = compute_hydrophobic_face_angle(seq, list(range(L)))

    # Hydrophobic vs PositiveFace 角度差
    hyd_angles_deg = [(i * DEGREES_PER_RESIDUE) % 360.0 for i in hyd_lin]
    hyd_face_angle = hyd_angles_deg[-1] - hyd_angles_deg[0] if hyd_angles_deg else 0.0

    feats['Hydrophobic_vs_CoreV3_PosFaceAngleDeg'] = circular_distance_deg(hyd_face_angle, feats['PositiveFaceAngleDeg_CoreV3'])
    feats['Hydrophobic_vs_Continuous_PosFaceAngleDeg'] = circular_distance_deg(hyd_face_angle, feats['PositiveFaceAngleDeg_Continuous'])

    # PositiveFace vs HydrophobicFace 角度差
    feats['Positive_vs_HydrophobicFaceAngleDeg'] = circular_distance_deg(feats['PositiveFaceAngleDeg_CoreV3'], feats['HydrophobicFaceAngleDeg_CoreV3'])

    return feats
