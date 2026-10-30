"""Benchmark evaluation (Protein-Held-Out GroupKFold vs Random KFold) and Kaggle submission generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from enzstab.features import FEATURE_COLUMNS


def _norm_rank(arr: np.ndarray) -> np.ndarray:
    """Return percentile rank in (0, 1]."""
    return rankdata(arr, method="average") / float(len(arr))


def compute_zero_shot_biophysical_score(df: pd.DataFrame) -> np.ndarray:
    """Compute the Kaggle Novozymes zero-shot biophysical rank composite (BLOSUM62 - pLDDT - 3D packing - core clash)."""
    r_b62 = _norm_rank(df["blosum62"].to_numpy(dtype=np.float64))
    r_plddt = _norm_rank(-df["af_plddt"].to_numpy(dtype=np.float64))
    r_pack = _norm_rank(-df["ca_packing_10a"].to_numpy(dtype=np.float64))
    r_clash = _norm_rank(-df["core_hydrophobic_clash"].to_numpy(dtype=np.float64))
    r_vol = _norm_rank(-df["abs_delta_volume"].to_numpy(dtype=np.float64))
    return 0.30 * r_b62 + 0.25 * r_plddt + 0.20 * r_pack + 0.15 * r_clash + 0.10 * r_vol


def _evaluate_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    model_name: str,
    split_scheme: str,
) -> dict[str, Any]:
    """Compute global Spearman rho, mean within-protein Spearman rho, RMSE, and ROC-AUC."""
    y_rank = df["within_protein_rank"].to_numpy(dtype=np.float64)
    y_raw = df["stability_value"].to_numpy(dtype=np.float64)
    y_bin = df["is_stabilizing"].to_numpy(dtype=np.int64)

    global_rho_rank = float(spearmanr(y_rank, preds).statistic)
    global_rho_raw = float(spearmanr(y_raw, preds).statistic)
    rank_rmse = float(np.sqrt(mean_squared_error(y_rank, preds)))
    roc_auc = float(roc_auc_score(y_bin, preds))

    # Mean within-protein Spearman rho (exact Kaggle Novozymes evaluation regime)
    prot_rhos: list[float] = []
    ddg_rhos: list[float] = []
    dtm_rhos: list[float] = []
    for _, grp in df.groupby("uniprot_id", sort=False):
        if len(grp) < 5:
            continue
        p_idx = grp.index.to_numpy()
        r_val = float(spearmanr(grp["stability_value"].to_numpy(), preds[p_idx]).statistic)
        if not np.isnan(r_val):
            prot_rhos.append(r_val)
            if grp["assay_type"].iloc[0] == "ddG":
                ddg_rhos.append(r_val)
            else:
                dtm_rhos.append(r_val)

    return {
        "model": model_name,
        "split_scheme": split_scheme,
        "mean_protein_spearman": round(float(np.mean(prot_rhos)), 4),
        "global_spearman_rank": round(global_rho_rank, 4),
        "global_spearman_raw": round(global_rho_raw, 4),
        "ddg_cohort_spearman": round(float(np.mean(ddg_rhos)), 4),
        "dtm_cohort_spearman": round(float(np.mean(dtm_rhos)), 4),
        "rank_rmse": round(rank_rmse, 4),
        "stabilizing_roc_auc": round(roc_auc, 4),
    }


def _build_estimators() -> dict[str, Any]:
    """Instantiate the three supervised regressors."""
    return {
        "Ridge Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=60.0, random_state=42)),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            max_depth=4,
            learning_rate=0.035,
            min_samples_leaf=22,
            l2_regularization=3.0,
            random_state=42,
        ),
    }


def run_benchmark_evaluation(
    df: pd.DataFrame,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run single-feature baselines, 5-fold Protein-Held-Out GroupKFold, and 5-fold Random KFold."""
    results_dir.mkdir(parents=True, exist_ok=True)
    work_df = df.reset_index(drop=True).copy()
    X = work_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = work_df["within_protein_rank"].to_numpy(dtype=np.float64)
    groups = work_df["uniprot_id"].to_numpy()

    metrics_rows: list[dict[str, Any]] = []

    # 1. Single-feature & zero-shot biophysical baselines (evaluated on the full protein-held-out cohort)
    baselines = {
        "BLOSUM62 Baseline": _norm_rank(work_df["blosum62"].to_numpy(dtype=np.float64)),
        "AlphaFold -pLDDT Baseline": _norm_rank(-work_df["af_plddt"].to_numpy(dtype=np.float64)),
        "3D -Packing (10A) Baseline": _norm_rank(-work_df["ca_packing_10a"].to_numpy(dtype=np.float64)),
        "Zero-Shot Biophysical Rank": compute_zero_shot_biophysical_score(work_df),
    }
    for b_name, b_preds in baselines.items():
        metrics_rows.append(
            _evaluate_predictions(work_df, b_preds, b_name, "Zero-Shot (Protein-Agnostic)")
        )

    # 2. Supervised 5-fold Protein-Held-Out GroupKFold (0 shared proteins between train & test)
    gkf = GroupKFold(n_splits=5)
    oof_group: dict[str, np.ndarray] = {
        name: np.zeros(len(work_df), dtype=np.float64) for name in _build_estimators()
    }
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        for name, est in _build_estimators().items():
            est.fit(X[train_idx], y[train_idx])
            oof_group[name][val_idx] = np.clip(est.predict(X[val_idx]), 0.0, 1.0)

    for name, preds in oof_group.items():
        metrics_rows.append(
            _evaluate_predictions(work_df, preds, name, "GroupKFold (Protein-Held-Out)")
        )

    # Kaggle Grandmaster Rank-Ensemble (combining OOF ML models + zero-shot biophysical prior)
    zs_score = baselines["Zero-Shot Biophysical Rank"]
    ens_group = (
        0.25 * _norm_rank(oof_group["Ridge Regression"])
        + 0.25 * _norm_rank(oof_group["Random Forest"])
        + 0.30 * _norm_rank(oof_group["HistGradientBoosting"])
        + 0.20 * _norm_rank(zs_score)
    )
    metrics_rows.append(
        _evaluate_predictions(
            work_df,
            ens_group,
            "Kaggle Rank-Average Ensemble",
            "GroupKFold (Protein-Held-Out)",
        )
    )

    # 3. Supervised 5-fold Random KFold (demonstrating protein-identity leakage across shared proteins)
    rkf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_rand: dict[str, np.ndarray] = {
        name: np.zeros(len(work_df), dtype=np.float64) for name in _build_estimators()
    }
    shared_proteins_per_fold: list[int] = []
    for train_idx, val_idx in rkf.split(X, y):
        shared_proteins_per_fold.append(
            len(set(groups[train_idx]).intersection(set(groups[val_idx])))
        )
        for name, est in _build_estimators().items():
            est.fit(X[train_idx], y[train_idx])
            oof_rand[name][val_idx] = np.clip(est.predict(X[val_idx]), 0.0, 1.0)

    for name, preds in oof_rand.items():
        metrics_rows.append(
            _evaluate_predictions(work_df, preds, name, "Random KFold (Protein Leakage)")
        )

    ens_rand = (
        0.25 * _norm_rank(oof_rand["Ridge Regression"])
        + 0.25 * _norm_rank(oof_rand["Random Forest"])
        + 0.30 * _norm_rank(oof_rand["HistGradientBoosting"])
        + 0.20 * _norm_rank(zs_score)
    )
    metrics_rows.append(
        _evaluate_predictions(
            work_df,
            ens_rand,
            "Kaggle Rank-Average Ensemble",
            "Random KFold (Protein Leakage)",
        )
    )

    # Save out-of-fold predictions
    work_df["oof_ridge_group"] = np.round(oof_group["Ridge Regression"], 4)
    work_df["oof_rf_group"] = np.round(oof_group["Random Forest"], 4)
    work_df["oof_hgb_group"] = np.round(oof_group["HistGradientBoosting"], 4)
    work_df["oof_ensemble_group"] = np.round(ens_group, 4)
    work_df["oof_ensemble_random"] = np.round(ens_rand, 4)
    work_df.to_csv(results_dir / "oof_predictions.csv", index=False)

    # Per-protein breakdown table
    per_prot_rows: list[dict[str, Any]] = []
    for uniprot_id, grp in work_df.groupby("uniprot_id", sort=False):
        p_idx = grp.index.to_numpy()
        y_p = grp["stability_value"].to_numpy()
        per_prot_rows.append(
            {
                "pdb_id": grp["pdb_id"].iloc[0],
                "uniprot_id": uniprot_id,
                "protein_name": grp["protein_name"].iloc[0],
                "organism": grp["organism"].iloc[0],
                "assay_type": grp["assay_type"].iloc[0],
                "n_mutations": len(grp),
                "mean_af_plddt": round(float(grp["af_plddt"].mean()), 2),
                "spearman_blosum62": round(float(spearmanr(y_p, baselines["BLOSUM62 Baseline"][p_idx]).statistic), 4),
                "spearman_neg_plddt": round(float(spearmanr(y_p, baselines["AlphaFold -pLDDT Baseline"][p_idx]).statistic), 4),
                "spearman_hgb_group": round(float(spearmanr(y_p, oof_group["HistGradientBoosting"][p_idx]).statistic), 4),
                "spearman_ensemble_group": round(float(spearmanr(y_p, ens_group[p_idx]).statistic), 4),
                "spearman_ensemble_random": round(float(spearmanr(y_p, ens_rand[p_idx]).statistic), 4),
            }
        )
    per_prot_df = pd.DataFrame(per_prot_rows)
    per_prot_df.to_csv(results_dir / "per_protein_spearman.csv", index=False)

    # Permutation feature importance on full HistGradientBoosting fit
    full_hgb = _build_estimators()["HistGradientBoosting"]
    full_hgb.fit(X, y)
    perm = permutation_importance(
        full_hgb, X, y, n_repeats=10, random_state=42, scoring="r2"
    )
    fi_df = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": np.round(perm.importances_mean, 5),
                "importance_std": np.round(perm.importances_std, 5),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    fi_df.to_csv(results_dir / "feature_importance.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(results_dir / "benchmark_metrics.csv", index=False)

    summary = {
        "n_mutations": len(work_df),
        "n_proteins": int(work_df["uniprot_id"].nunique()),
        "n_ddg_mutations": int((work_df["assay_type"] == "ddG").sum()),
        "n_dtm_mutations": int((work_df["assay_type"] == "dTm").sum()),
        "n_ddg_proteins": int(work_df.loc[work_df["assay_type"] == "ddG", "uniprot_id"].nunique()),
        "n_dtm_proteins": int(work_df.loc[work_df["assay_type"] == "dTm", "uniprot_id"].nunique()),
        "shared_proteins_group_kfold": 0,
        "mean_shared_proteins_random_kfold": round(float(np.mean(shared_proteins_per_fold)), 1),
        "metrics": metrics_rows,
        "top_features": fi_df.head(10).to_dict(orient="records"),
    }
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return metrics_df, fi_df, summary


def generate_kaggle_submission(
    train_df: pd.DataFrame,
    kaggle_test_df: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    """Train final models on the 36-protein benchmark and generate the 2,413-variant Kaggle submission.csv."""
    results_dir.mkdir(parents=True, exist_ok=True)
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = train_df["within_protein_rank"].to_numpy(dtype=np.float64)
    X_test = kaggle_test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    estimators = _build_estimators()
    test_preds: dict[str, np.ndarray] = {}
    for name, est in estimators.items():
        est.fit(X_train, y_train)
        test_preds[name] = est.predict(X_test)

    zs_test = compute_zero_shot_biophysical_score(kaggle_test_df)
    ensemble_rank = (
        0.25 * _norm_rank(test_preds["Ridge Regression"])
        + 0.25 * _norm_rank(test_preds["Random Forest"])
        + 0.30 * _norm_rank(test_preds["HistGradientBoosting"])
        + 0.20 * _norm_rank(zs_test)
    )

    # Penalize single-amino-acid deletions to the bottom decile (standard Novozymes post-processing)
    is_del = (kaggle_test_df["edit_type"] == "deletion").to_numpy()
    final_tm_rank = ensemble_rank.copy()
    final_tm_rank[is_del] *= 0.25
    final_tm_rank = _norm_rank(final_tm_rank)

    sub_df = pd.DataFrame(
        {
            "seq_id": kaggle_test_df["seq_id"].astype(int),
            "tm": np.round(final_tm_rank, 6),
        }
    )
    sub_df.to_csv(results_dir / "submission.csv", index=False)

    detailed_df = kaggle_test_df[
        ["seq_id", "edit_type", "position", "wildtype", "mutation", "mutation_code", "pH", "af_plddt", "ca_packing_10a", "blosum62"]
    ].copy()
    detailed_df["pred_ridge_rank"] = np.round(_norm_rank(test_preds["Ridge Regression"]), 4)
    detailed_df["pred_rf_rank"] = np.round(_norm_rank(test_preds["Random Forest"]), 4)
    detailed_df["pred_hgb_rank"] = np.round(_norm_rank(test_preds["HistGradientBoosting"]), 4)
    detailed_df["pred_zeroshot_rank"] = np.round(_norm_rank(zs_test), 4)
    detailed_df["tm"] = np.round(final_tm_rank, 6)
    detailed_df.to_csv(results_dir / "kaggle_test_ranked_variants.csv", index=False)

    return sub_df
