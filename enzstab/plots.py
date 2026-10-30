"""Generate high-resolution benchmark figures for Kaggle Novozymes & AlphaFold Thermostability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 160,
    }
)


def plot_spearman_benchmark(metrics_df: pd.DataFrame, out_path: Path) -> Path:
    """Plot Spearman rank correlation and ROC-AUC across zero-shot, GroupKFold, and Random KFold."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Zero-shot baselines vs GroupKFold (Protein-Held-Out)
    ax1 = axes[0]
    sub = metrics_df[
        metrics_df["split_scheme"].isin(
            ["Zero-Shot (Protein-Agnostic)", "GroupKFold (Protein-Held-Out)"]
        )
    ].copy()
    labels = [
        f"{r['model']}\n({'Zero-Shot' if 'Zero-Shot' in r['split_scheme'] else 'GroupKFold'})"
        for _, r in sub.iterrows()
    ]
    y_pos = np.arange(len(sub))
    h = 0.36

    b1 = ax1.barh(
        y_pos + h / 2,
        sub["global_spearman_rank"],
        height=h,
        color="#1d4ed8",
        label="Global Spearman ρ (Rank Target)",
    )
    b2 = ax1.barh(
        y_pos - h / 2,
        sub["mean_protein_spearman"],
        height=h,
        color="#0d9488",
        label="Mean Within-Protein Spearman ρ",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 0.62)
    ax1.set_xlabel("Spearman Rank Correlation (ρ)")
    ax1.set_title("A. Protein-Held-Out Benchmark (36 Proteins, n = 2,766)")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.5)

    for bar in list(b1) + list(b2):
        w = bar.get_width()
        ax1.text(
            w + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Panel B: Protein-Held-Out GroupKFold vs Protein-Leaking Random KFold
    ax2 = axes[1]
    ml_models = [
        "Ridge Regression",
        "Random Forest",
        "HistGradientBoosting",
        "Kaggle Rank-Average Ensemble",
    ]
    g_sub = (
        metrics_df[metrics_df["split_scheme"] == "GroupKFold (Protein-Held-Out)"]
        .set_index("model")
        .loc[ml_models]
    )
    r_sub = (
        metrics_df[metrics_df["split_scheme"] == "Random KFold (Protein Leakage)"]
        .set_index("model")
        .loc[ml_models]
    )

    x = np.arange(len(ml_models))
    w = 0.35
    bars_g = ax2.bar(
        x - w / 2,
        g_sub["global_spearman_rank"],
        width=w,
        color="#2563eb",
        label="GroupKFold (0 Shared Proteins)",
    )
    bars_r = ax2.bar(
        x + w / 2,
        r_sub["global_spearman_rank"],
        width=w,
        color="#f59e0b",
        label="Random KFold (36 Shared Proteins)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        ["Ridge", "Random\nForest", "HistGrad\nBoosting", "Kaggle Rank\nEnsemble"],
        fontsize=9,
    )
    ax2.set_ylim(0, 0.65)
    ax2.set_ylabel("Global Spearman Correlation (ρ)")
    ax2.set_title("B. GroupKFold vs Random Split Leakage")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8.5)

    for bar in list(bars_g) + list(bars_r):
        val = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_plddt_vs_stability(df: pd.DataFrame, out_path: Path) -> Path:
    """Plot AlphaFold pLDDT and 3D 10A C-alpha packing vs experimental enzyme thermostability."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Panel A: AlphaFold pLDDT confidence regime vs Stability Rank & Stabilizing Rate
    ax1 = axes[0]
    bins = [0, 75, 85, 92, 96, 100.1]
    bin_labels = [
        "Loop / Flexible\n(pLDDT < 75)",
        "Moderate\n(75–85)",
        "Ordered\n(85–92)",
        "High Conf.\n(92–96)",
        "Ultra-Core\n(96–100)",
    ]
    work = df.copy()
    work["plddt_bin"] = pd.cut(work["af_plddt"], bins=bins, labels=bin_labels, right=False)
    grp = work.groupby("plddt_bin", observed=False).agg(
        mean_rank=("within_protein_rank", "mean"),
        sem_rank=("within_protein_rank", "sem"),
        stab_rate=("is_stabilizing", "mean"),
        count=("within_protein_rank", "count"),
    )

    x = np.arange(len(grp))
    bars = ax1.bar(
        x,
        grp["mean_rank"],
        yerr=grp["sem_rank"],
        capsize=4,
        color=["#10b981", "#34d399", "#60a5fa", "#3b82f6", "#1e3a8a"],
        edgecolor="#1f2937",
        width=0.58,
    )
    ax1.axhline(0.50, color="#dc2626", linestyle="--", linewidth=1.2, label="Neutral Rank (0.50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grp.index, fontsize=8.8)
    ax1.set_ylim(0, 0.92)
    ax1.set_ylabel("Mean Within-Protein Thermostability Rank")
    ax1.set_title("A. AlphaFold pLDDT vs Mutation Thermostability")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        cnt = int(grp["count"].iloc[i])
        sr = 100.0 * float(grp["stab_rate"].iloc[i])
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.028,
            f"{h:.2f}\n({sr:.0f}% stab, n={cnt})",
            ha="center",
            fontsize=7.8,
            fontweight="bold",
        )

    # Panel B: 3D C-alpha Packing Density (10A) vs Stability across BLOSUM62 tiers
    ax2 = axes[1]
    work["packing_tier"] = pd.cut(
        work["ca_packing_10a"],
        bins=[-1, 12, 17, 22, 50],
        labels=[
            "Exposed Surface\n(≤12 neighbors)",
            "Intermediate\n(13–17 neighbors)",
            "Buried Core\n(18–22 neighbors)",
            "Dense Hydrophobic Core\n(≥23 neighbors)",
        ],
    )
    work["blosum_tier"] = np.where(
        work["blosum62"] >= 0,
        "Conservative (BLOSUM62 ≥ 0)",
        "Radical Substitution (BLOSUM62 < 0)",
    )
    piv = work.pivot_table(
        index="packing_tier",
        columns="blosum_tier",
        values="within_protein_rank",
        aggfunc="mean",
        observed=False,
    )

    x2 = np.arange(len(piv))
    w = 0.36
    c1 = "Conservative (BLOSUM62 ≥ 0)"
    c2 = "Radical Substitution (BLOSUM62 < 0)"
    b_cons = ax2.bar(x2 - w / 2, piv[c1], width=w, color="#0ea5e9", label=c1)
    b_rad = ax2.bar(x2 + w / 2, piv[c2], width=w, color="#ef4444", label=c2)
    ax2.axhline(0.50, color="#6b7280", linestyle="--", linewidth=1.1)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(piv.index, fontsize=8.5)
    ax2.set_ylim(0, 0.78)
    ax2.set_ylabel("Mean Within-Protein Thermostability Rank")
    ax2.set_title("B. 3D Cα Packing (10 Å) × Evolutionary Conservation")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)

    for bar in list(b_cons) + list(b_rad):
        val = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.012,
            f"{val:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(
    fi_df: pd.DataFrame,
    per_prot_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Plot permutation feature importance and per-protein Spearman correlation across 36 enzymes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.5), gridspec_kw={"width_ratios": [1.1, 1.2]})

    # Panel A: Top 12 features by permutation importance
    ax1 = axes[0]
    top_fi = fi_df.head(12).iloc[::-1]
    y_pos = np.arange(len(top_fi))
    bars = ax1.barh(
        y_pos,
        top_fi["importance_mean"],
        xerr=top_fi["importance_std"],
        capsize=3,
        color="#4f46e5",
        alpha=0.9,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_fi["feature"], fontsize=9)
    ax1.set_xlabel("Permutation Importance (ΔR² on Rank Target)")
    ax1.set_title("A. Top 12 Sequence + 3D AlphaFold Features")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax1.text(
            w + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=8,
        )
    ax1.set_xlim(0, float(top_fi["importance_mean"].max()) * 1.22)

    # Panel B: Top 15 largest protein cohorts protein-held-out Spearman rho
    ax2 = axes[1]
    top_prots = per_prot_df.sort_values("n_mutations", ascending=False).head(15).iloc[::-1]
    y2 = np.arange(len(top_prots))
    colors = [
        "#0d9488" if a == "dTm" else "#2563eb" for a in top_prots["assay_type"]
    ]
    bars2 = ax2.barh(y2, top_prots["spearman_ensemble_group"], color=colors, alpha=0.9)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(
        [
            f"{r['pdb_id']} ({r['uniprot_id']}, n={r['n_mutations']}, {r['assay_type']})"
            for _, r in top_prots.iterrows()
        ],
        fontsize=8.5,
    )
    ax2.set_xlabel("Held-Out Protein Spearman Rank Correlation (ρ)")
    ax2.set_title("B. Held-Out Spearman ρ Across Major Enzyme Cohorts")
    ax2.axvline(
        float(per_prot_df["spearman_ensemble_group"].mean()),
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"36-Protein Mean ρ = {per_prot_df['spearman_ensemble_group'].mean():.3f}",
    )
    ax2.set_xlim(0, 0.78)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=8.5)

    for bar in bars2:
        w = bar.get_width()
        ax2.text(
            max(w + 0.01, 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
