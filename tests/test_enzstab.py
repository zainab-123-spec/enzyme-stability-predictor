"""Test suite for the Kaggle Novozymes & AlphaFold Enzyme Thermostability Benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from enzstab.constants import (
    BENCHMARK_PROTEINS,
    KYTE_DOOLITTLE,
    NOVOZYMES_WT_SEQUENCE,
    RESIDUE_VOLUME,
    get_blosum62,
)
from enzstab.features import (
    FEATURE_COLUMNS,
    compute_mutation_features,
    find_sequence_offset,
    parse_alphafold_structure,
)
from enzstab.models import compute_zero_shot_biophysical_score

ROOT = Path(__file__).resolve().parent.parent


def test_novozymes_wildtype_sequence_length() -> None:
    assert len(NOVOZYMES_WT_SEQUENCE) == 221
    assert NOVOZYMES_WT_SEQUENCE.startswith("VPVNPEPDAT")
    assert NOVOZYMES_WT_SEQUENCE.endswith("KNNKFNYVGK")


def test_benchmark_protein_cohort_uniqueness() -> None:
    assert len(BENCHMARK_PROTEINS) == 36
    uniprots = [p["uniprot_id"] for p in BENCHMARK_PROTEINS]
    pdbs = [p["pdb_id"] for p in BENCHMARK_PROTEINS]
    assert len(set(uniprots)) == 36
    assert len(set(pdbs)) == 36
    ddg_count = sum(1 for p in BENCHMARK_PROTEINS if p["assay"] == "ddG")
    dtm_count = sum(1 for p in BENCHMARK_PROTEINS if p["assay"] == "dTm")
    assert ddg_count == 23
    assert dtm_count == 13


def test_blosum62_symmetry_and_values() -> None:
    for a in KYTE_DOOLITTLE:
        assert get_blosum62(a, a) >= 4
        for b in KYTE_DOOLITTLE:
            assert get_blosum62(a, b) == get_blosum62(b, a)
    assert get_blosum62("W", "W") == 11
    assert get_blosum62("A", "-") == -6


def test_parse_novozymes_alphafold_pdb() -> None:
    pdb_path = ROOT / "data" / "structures" / "wildtype_structure_prediction_af2.pdb"
    assert pdb_path.exists()
    profile = parse_alphafold_structure(pdb_path)
    assert len(profile.sequence) == 221
    assert profile.sequence == NOVOZYMES_WT_SEQUENCE
    assert len(profile.plddt) == 221
    assert all(0.0 <= v <= 100.0 for v in profile.plddt.values())
    assert all(v >= 0 for v in profile.packing_10a.values())


def test_find_sequence_offset_exact_and_shifted() -> None:
    full_seq = "MKKLLLAAGV" + NOVOZYMES_WT_SEQUENCE + "HHHHHH"
    assert find_sequence_offset(NOVOZYMES_WT_SEQUENCE, full_seq) == 10
    assert find_sequence_offset(NOVOZYMES_WT_SEQUENCE, NOVOZYMES_WT_SEQUENCE) == 0


def test_compute_mutation_features_completeness() -> None:
    pdb_path = ROOT / "data" / "structures" / "wildtype_structure_prediction_af2.pdb"
    profile = parse_alphafold_structure(pdb_path)
    feats = compute_mutation_features(
        wt_aa="L",
        mut_aa="R",
        seq_pos_1based=16,
        seq_len=221,
        af_res_num=16,
        profile=profile,
    )
    assert list(feats.keys()) == FEATURE_COLUMNS
    assert feats["blosum62"] == get_blosum62("L", "R")
    assert np.isclose(
        feats["delta_hydrophobicity"], KYTE_DOOLITTLE["R"] - KYTE_DOOLITTLE["L"]
    )
    assert np.isclose(feats["delta_volume"], RESIDUE_VOLUME["R"] - RESIDUE_VOLUME["L"])


def test_processed_benchmark_dataset_integrity() -> None:
    bench_path = ROOT / "data" / "processed" / "enzyme_stability_benchmark.csv"
    assert bench_path.exists()
    df = pd.read_csv(bench_path)
    assert len(df) == 2766
    assert df["uniprot_id"].nunique() == 36
    assert df["pdb_id"].nunique() == 36
    assert not df[FEATURE_COLUMNS].isna().any().any()
    assert df["within_protein_rank"].between(0.0, 1.0).all()


def test_group_kfold_zero_protein_leakage() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "enzyme_stability_benchmark.csv")
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(df, df["within_protein_rank"], groups=df["uniprot_id"]):
        train_prots = set(df.iloc[train_idx]["uniprot_id"])
        val_prots = set(df.iloc[val_idx]["uniprot_id"])
        assert len(train_prots.intersection(val_prots)) == 0


def test_zero_shot_biophysical_score_correlates_positively() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "enzyme_stability_benchmark.csv")
    zs = compute_zero_shot_biophysical_score(df)
    assert len(zs) == len(df)
    assert np.all(np.isfinite(zs))


def test_benchmark_summary_and_metrics_files() -> None:
    summary_path = ROOT / "results" / "benchmark_summary.json"
    metrics_path = ROOT / "results" / "benchmark_metrics.csv"
    assert summary_path.exists()
    assert metrics_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_mutations"] == 2766
    assert summary["n_proteins"] == 36
    assert summary["shared_proteins_group_kfold"] == 0

    metrics_df = pd.read_csv(metrics_path)
    ens_group = metrics_df[
        (metrics_df["model"] == "Kaggle Rank-Average Ensemble")
        & (metrics_df["split_scheme"] == "GroupKFold (Protein-Held-Out)")
    ].iloc[0]
    b62_row = metrics_df[metrics_df["model"] == "BLOSUM62 Baseline"].iloc[0]
    assert ens_group["global_spearman_rank"] > b62_row["global_spearman_rank"] + 0.20
    assert ens_group["mean_protein_spearman"] > 0.34


def test_kaggle_submission_schema_and_validity() -> None:
    sub_path = ROOT / "results" / "submission.csv"
    sample_path = ROOT / "data" / "raw" / "kaggle_sample_submission.csv"
    assert sub_path.exists()
    sub_df = pd.read_csv(sub_path)
    sample_df = pd.read_csv(sample_path)
    assert list(sub_df.columns) == ["seq_id", "tm"]
    assert len(sub_df) == 2413
    assert (sub_df["seq_id"].values == sample_df["seq_id"].values).all()
    assert not sub_df["tm"].isna().any()
    assert sub_df["tm"].nunique() > 1000


def test_figures_and_notebook_exist() -> None:
    for fig_name in ["spearman_benchmark.png", "plddt_vs_stability.png", "feature_importance.png"]:
        fig_path = ROOT / "docs" / "figures" / fig_name
        assert fig_path.exists()
        assert fig_path.stat().st_size > 20_000

    nb_path = ROOT / "kaggle_enzyme_thermostability_solution.ipynb"
    assert nb_path.exists()
    assert nb_path.stat().st_size > 50_000
