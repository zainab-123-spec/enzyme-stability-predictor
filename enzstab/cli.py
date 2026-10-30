"""Command-line interface for the Kaggle Novozymes & AlphaFold Thermostability Benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from enzstab.features import build_benchmark_dataset, build_kaggle_test_features
from enzstab.fetch import fetch_all_raw_datasets
from enzstab.models import generate_kaggle_submission, run_benchmark_evaluation
from enzstab.plots import (
    plot_feature_importance,
    plot_plddt_vs_stability,
    plot_spearman_benchmark,
)


def run_all(root_dir: Path) -> None:
    """Execute the complete fetch, feature extraction, evaluation, plotting, and submission pipeline."""
    data_dir = root_dir / "data"
    results_dir = root_dir / "results"
    fig_dir = root_dir / "docs" / "figures"

    print("[1/5] Fetching ProTherm/FireProt mutations, AlphaFold DB v6 structures, UniProt/RCSB metadata, and Kaggle Novozymes files...")
    fetch_all_raw_datasets(data_dir)

    print("[2/5] Extracting 19 sequence + 3D AlphaFold features for 36 benchmark proteins and 2,413 Kaggle test variants...")
    bench_df = build_benchmark_dataset(data_dir)
    kaggle_test_df = build_kaggle_test_features(data_dir)

    print("[3/5] Running 5-fold Protein-Held-Out GroupKFold vs Random KFold benchmark evaluation...")
    metrics_df, fi_df, summary = run_benchmark_evaluation(bench_df, results_dir)

    print("[4/5] Generating official Kaggle Novozymes submission.csv (n = 2,413 variants)...")
    sub_df = generate_kaggle_submission(bench_df, kaggle_test_df, results_dir)

    print("[5/5] Rendering publication figures in docs/figures/...")
    per_prot_df = pd.read_csv(results_dir / "per_protein_spearman.csv")
    plot_spearman_benchmark(metrics_df, fig_dir / "spearman_benchmark.png")
    plot_plddt_vs_stability(bench_df, fig_dir / "plddt_vs_stability.png")
    plot_feature_importance(fi_df, per_prot_df, fig_dir / "feature_importance.png")

    print("\n=== BENCHMARK COMPLETE ===")
    print(f"Benchmark cohort: {summary['n_mutations']} mutations across {summary['n_proteins']} proteins")
    print(f"Kaggle submission: {len(sub_df)} rows written to {results_dir / 'submission.csv'}")
    print("\nSummary Metrics:")
    print(metrics_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle Novozymes & AlphaFold Enzyme Thermostability Benchmark CLI"
    )
    parser.add_argument(
        "command",
        choices=["all", "fetch", "features", "evaluate", "submit", "plots"],
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root directory",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    if args.command == "all":
        run_all(root)
    elif args.command == "fetch":
        fetch_all_raw_datasets(root / "data")
    elif args.command == "features":
        build_benchmark_dataset(root / "data")
        build_kaggle_test_features(root / "data")
    elif args.command == "evaluate":
        bench_df = pd.read_csv(root / "data" / "processed" / "enzyme_stability_benchmark.csv")
        run_benchmark_evaluation(bench_df, root / "results")
    elif args.command == "submit":
        bench_df = pd.read_csv(root / "data" / "processed" / "enzyme_stability_benchmark.csv")
        test_df = pd.read_csv(root / "data" / "processed" / "kaggle_novozymes_test_features.csv")
        generate_kaggle_submission(bench_df, test_df, root / "results")
    elif args.command == "plots":
        bench_df = pd.read_csv(root / "data" / "processed" / "enzyme_stability_benchmark.csv")
        metrics_df = pd.read_csv(root / "results" / "benchmark_metrics.csv")
        fi_df = pd.read_csv(root / "results" / "feature_importance.csv")
        per_prot_df = pd.read_csv(root / "results" / "per_protein_spearman.csv")
        fig_dir = root / "docs" / "figures"
        plot_spearman_benchmark(metrics_df, fig_dir / "spearman_benchmark.png")
        plot_plddt_vs_stability(bench_df, fig_dir / "plddt_vs_stability.png")
        plot_feature_importance(fi_df, per_prot_df, fig_dir / "feature_importance.png")


if __name__ == "__main__":
    main()
