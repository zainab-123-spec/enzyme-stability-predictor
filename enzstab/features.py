"""Sequence and 3D AlphaFold structural feature extraction for enzyme thermostability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from enzstab.constants import (
    AA3_TO_1,
    BENCHMARK_PROTEINS,
    HELIX_PROPENSITY,
    KYTE_DOOLITTLE,
    NOVOZYMES_WT_SEQUENCE,
    RESIDUE_CHARGE,
    RESIDUE_VOLUME,
    SHEET_PROPENSITY,
    get_blosum62,
)

FEATURE_COLUMNS: list[str] = [
    "af_plddt",
    "af_window_plddt",
    "ca_packing_10a",
    "ca_packing_15a",
    "rel_dist_center",
    "local_backbone_bend",
    "blosum62",
    "delta_hydrophobicity",
    "core_hydrophobic_clash",
    "delta_volume",
    "abs_delta_volume",
    "delta_abs_charge",
    "delta_helix_prop",
    "delta_sheet_prop",
    "is_proline_mutation",
    "is_glycine_mutation",
    "rel_seq_pos",
    "wt_hydrophobicity",
    "wt_volume",
]


@dataclass
class StructureProfile:
    """Precomputed 3D C-alpha coordinates and local environment metrics from an AlphaFold PDB."""

    sequence: str
    plddt: dict[int, float]
    window_plddt: dict[int, float]
    packing_10a: dict[int, int]
    packing_15a: dict[int, int]
    rel_dist_center: dict[int, float]
    backbone_bend: dict[int, float]


def parse_alphafold_structure(pdb_path: Path) -> StructureProfile:
    """Parse an AlphaFold PDB file and compute 3D C-alpha packing, burial, and pLDDT profiles."""
    residues: dict[int, tuple[str, np.ndarray, float]] = {}
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()
            aa = AA3_TO_1.get(res_name, "X")
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
            plddt_val = float(line[60:66].strip())
            residues[res_num] = (aa, xyz, plddt_val)

    sorted_nums = sorted(residues.keys())
    seq = "".join(residues[k][0] for k in sorted_nums)
    coords = np.vstack([residues[k][1] for k in sorted_nums])
    plddts = np.array([residues[k][2] for k in sorted_nums], dtype=np.float64)

    # Center of mass and radius of gyration
    center = coords.mean(axis=0)
    dists_to_center = np.linalg.norm(coords - center, axis=1)
    rg = float(np.sqrt(np.mean(dists_to_center**2))) or 1.0
    rel_dists = dists_to_center / rg

    # Pairwise C-alpha distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    pack_10 = (dist_mat <= 10.0).sum(axis=1)
    pack_15 = (dist_mat <= 15.0).sum(axis=1)

    plddt_map: dict[int, float] = {}
    win_plddt_map: dict[int, float] = {}
    pack10_map: dict[int, int] = {}
    pack15_map: dict[int, int] = {}
    rel_dist_map: dict[int, float] = {}
    bend_map: dict[int, float] = {}

    n_res = len(sorted_nums)
    for idx, res_num in enumerate(sorted_nums):
        plddt_map[res_num] = float(plddts[idx])
        lo = max(0, idx - 3)
        hi = min(n_res, idx + 4)
        win_plddt_map[res_num] = float(plddts[lo:hi].mean())
        pack10_map[res_num] = int(pack_10[idx])
        pack15_map[res_num] = int(pack_15[idx])
        rel_dist_map[res_num] = float(rel_dists[idx])

        if 0 < idx < n_res - 1:
            v1 = coords[idx - 1] - coords[idx]
            v2 = coords[idx + 1] - coords[idx]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = float(np.dot(v1, v2) / denom) if denom > 1e-6 else -0.5
            bend_map[res_num] = float(np.clip(cos_theta, -1.0, 1.0))
        else:
            bend_map[res_num] = -0.5

    return StructureProfile(
        sequence=seq,
        plddt=plddt_map,
        window_plddt=win_plddt_map,
        packing_10a=pack10_map,
        packing_15a=pack15_map,
        rel_dist_center=rel_dist_map,
        backbone_bend=bend_map,
    )


def find_sequence_offset(construct_seq: str, af_seq: str) -> int:
    """Find the 0-based offset aligning a PDB construct sequence into its full-length AlphaFold sequence."""
    idx = af_seq.find(construct_seq)
    if idx != -1:
        return idx

    best_off = 0
    best_matches = -1
    for off in range(-50, len(af_seq) - 10):
        matches = 0
        for i, ch in enumerate(construct_seq):
            j = off + i
            if 0 <= j < len(af_seq) and af_seq[j] == ch:
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_off = off
    return best_off


def compute_mutation_features(
    wt_aa: str,
    mut_aa: str,
    seq_pos_1based: int,
    seq_len: int,
    af_res_num: int,
    profile: StructureProfile,
) -> dict[str, float]:
    """Compute the 19 sequence + 3D structural features for a single mutation or deletion."""
    wt = str(wt_aa).upper()
    mut = str(mut_aa).upper()

    af_plddt = float(profile.plddt.get(af_res_num, 75.0))
    af_win_plddt = float(profile.window_plddt.get(af_res_num, af_plddt))
    pack_10 = float(profile.packing_10a.get(af_res_num, 12))
    pack_15 = float(profile.packing_15a.get(af_res_num, 35))
    rel_dist = float(profile.rel_dist_center.get(af_res_num, 1.0))
    bend = float(profile.backbone_bend.get(af_res_num, -0.5))

    wt_h = KYTE_DOOLITTLE.get(wt, 0.0)
    wt_v = RESIDUE_VOLUME.get(wt, 130.0)
    wt_q = RESIDUE_CHARGE.get(wt, 0.0)
    wt_helix = HELIX_PROPENSITY.get(wt, 1.0)
    wt_sheet = SHEET_PROPENSITY.get(wt, 1.0)

    if mut in KYTE_DOOLITTLE:
        b62 = float(get_blosum62(wt, mut))
        mut_h = KYTE_DOOLITTLE[mut]
        mut_v = RESIDUE_VOLUME[mut]
        mut_q = RESIDUE_CHARGE[mut]
        mut_helix = HELIX_PROPENSITY[mut]
        mut_sheet = SHEET_PROPENSITY[mut]
    else:
        # Deletion or non-standard edit
        b62 = -6.0
        mut_h = 0.0
        mut_v = 0.0
        mut_q = 0.0
        mut_helix = 0.5
        mut_sheet = 0.5

    delta_h = mut_h - wt_h
    delta_v = mut_v - wt_v
    abs_delta_v = abs(delta_v)
    delta_abs_q = abs(mut_q) - abs(wt_q)
    delta_helix = mut_helix - wt_helix
    delta_sheet = mut_sheet - wt_sheet
    core_clash = (pack_10 / 20.0) * abs(delta_h)
    is_pro = 1.0 if (wt == "P" or mut == "P") else 0.0
    is_gly = 1.0 if (wt == "G" or mut == "G") else 0.0
    rel_pos = float(seq_pos_1based) / float(max(seq_len, 1))

    return {
        "af_plddt": round(af_plddt, 4),
        "af_window_plddt": round(af_win_plddt, 4),
        "ca_packing_10a": round(pack_10, 4),
        "ca_packing_15a": round(pack_15, 4),
        "rel_dist_center": round(rel_dist, 4),
        "local_backbone_bend": round(bend, 4),
        "blosum62": round(b62, 4),
        "delta_hydrophobicity": round(delta_h, 4),
        "core_hydrophobic_clash": round(core_clash, 4),
        "delta_volume": round(delta_v, 4),
        "abs_delta_volume": round(abs_delta_v, 4),
        "delta_abs_charge": round(delta_abs_q, 4),
        "delta_helix_prop": round(delta_helix, 4),
        "delta_sheet_prop": round(delta_sheet, 4),
        "is_proline_mutation": is_pro,
        "is_glycine_mutation": is_gly,
        "rel_seq_pos": round(rel_pos, 4),
        "wt_hydrophobicity": round(wt_h, 4),
        "wt_volume": round(wt_v, 4),
    }


def build_benchmark_dataset(data_dir: Path) -> pd.DataFrame:
    """Build the full 36-protein experimental thermostability benchmark dataset with 3D AlphaFold features."""
    raw_dir = data_dir / "raw"
    struct_dir = data_dir / "structures"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    ddg_df = pd.read_csv(raw_dir / "protherm_ddg_raw.csv", index_col=0)
    dtm_df = pd.read_csv(raw_dir / "fireprot_dtm_raw.csv", index_col=0).rename(
        columns={"WT": "wildtype", "MUT": "mutation"}
    )

    rows: list[dict[str, float | int | str]] = []
    for p in BENCHMARK_PROTEINS:
        pdb_id = p["pdb_id"]
        uniprot_id = p["uniprot_id"]
        assay = p["assay"]
        pdb_path = struct_dir / f"AF-{uniprot_id}-F1.pdb"
        profile = parse_alphafold_structure(pdb_path)

        if assay == "ddG":
            sub = ddg_df[ddg_df["PDB"] == pdb_id].copy()
            target_col = "ddG"
        else:
            sub = dtm_df[dtm_df["PDB"] == pdb_id].copy()
            target_col = "dTm"

        if sub.empty:
            continue

        construct_seq = str(sub["sequence"].iloc[0])
        offset = find_sequence_offset(construct_seq, profile.sequence)

        # Compute within-protein percentile rank target in [0, 1] (as in Kaggle Grandmaster solutions)
        vals = sub[target_col].to_numpy(dtype=np.float64)
        pct_ranks = rankdata(vals, method="average") / float(len(vals))

        for (_, row), pct_rank in zip(sub.iterrows(), pct_ranks):
            pos = int(row["position"])
            wt = str(row["wildtype"]).upper()
            mut = str(row["mutation"]).upper()
            af_res_num = offset + pos
            exp_val = float(row[target_col])

            feats = compute_mutation_features(
                wt_aa=wt,
                mut_aa=mut,
                seq_pos_1based=pos,
                seq_len=len(construct_seq),
                af_res_num=af_res_num,
                profile=profile,
            )

            rec: dict[str, float | int | str] = {
                "pdb_id": pdb_id,
                "uniprot_id": uniprot_id,
                "protein_name": p["name"],
                "organism": p["organism"],
                "assay_type": assay,
                "position": pos,
                "af_residue_number": af_res_num,
                "wildtype": wt,
                "mutation": mut,
                "mutation_code": f"{wt}{pos}{mut}",
                "stability_value": round(exp_val, 4),
                "within_protein_rank": round(float(pct_rank), 4),
                "is_stabilizing": int(exp_val > 0.0),
            }
            rec.update(feats)
            rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(proc_dir / "enzyme_stability_benchmark.csv", index=False)
    return out_df


def _infer_novozymes_edit(seq: str, wt_seq: str = NOVOZYMES_WT_SEQUENCE) -> tuple[str, int, str, str]:
    """Identify the single-residue substitution or deletion in a Kaggle Novozymes test sequence."""
    if seq == wt_seq:
        return "wildtype", 1, wt_seq[0], wt_seq[0]

    if len(seq) == len(wt_seq):
        for i, (a, b) in enumerate(zip(wt_seq, seq)):
            if a != b:
                return "substitution", i + 1, a, b
        return "wildtype", 1, wt_seq[0], wt_seq[0]

    if len(seq) == len(wt_seq) - 1:
        for i, (a, b) in enumerate(zip(wt_seq, seq)):
            if a != b:
                return "deletion", i + 1, a, "-"
        return "deletion", len(wt_seq), wt_seq[-1], "-"

    return "substitution", 1, wt_seq[0], seq[0]


def build_kaggle_test_features(data_dir: Path) -> pd.DataFrame:
    """Extract the 19 sequence + AlphaFold 3D features for all 2,413 Kaggle Novozymes test variants."""
    raw_dir = data_dir / "raw"
    struct_dir = data_dir / "structures"
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(raw_dir / "kaggle_novozymes_test.csv")
    profile = parse_alphafold_structure(struct_dir / "wildtype_structure_prediction_af2.pdb")
    wt_len = len(NOVOZYMES_WT_SEQUENCE)

    rows: list[dict[str, float | int | str]] = []
    for _, row in test_df.iterrows():
        seq_id = int(row["seq_id"])
        seq = str(row["protein_sequence"])
        edit_type, pos, wt_aa, mut_aa = _infer_novozymes_edit(seq, NOVOZYMES_WT_SEQUENCE)
        feats = compute_mutation_features(
            wt_aa=wt_aa,
            mut_aa=mut_aa,
            seq_pos_1based=pos,
            seq_len=wt_len,
            af_res_num=pos,
            profile=profile,
        )
        rec: dict[str, float | int | str] = {
            "seq_id": seq_id,
            "edit_type": edit_type,
            "position": pos,
            "wildtype": wt_aa,
            "mutation": mut_aa,
            "mutation_code": f"{wt_aa}{pos}{mut_aa}",
            "pH": float(row["pH"]),
        }
        rec.update(feats)
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(proc_dir / "kaggle_novozymes_test_features.csv", index=False)
    return out_df
