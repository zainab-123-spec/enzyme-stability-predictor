"""Interactive Streamlit Explorer for Kaggle Novozymes & AlphaFold Enzyme Thermostability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from enzstab.constants import KYTE_DOOLITTLE, NOVOZYMES_WT_SEQUENCE
from enzstab.features import compute_mutation_features, parse_alphafold_structure

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "docs" / "figures"

st.set_page_config(
    page_title="Kaggle Novozymes & AlphaFold Thermostability Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Kaggle Novozymes Enzyme Stability & AlphaFold 3D Benchmark")
st.caption(
    "Predicting Enzyme Thermostability (ΔTm / ΔΔG) from Sequence + Live AlphaFold DB v6 3D Structures "
    "Across 36 Proteins (n = 2,766 Mutations) & 2,413 Kaggle Test Variants"
)

bench_df = pd.read_csv(DATA_DIR / "processed" / "enzyme_stability_benchmark.csv")
metrics_df = pd.read_csv(RESULTS_DIR / "benchmark_metrics.csv")
per_prot_df = pd.read_csv(RESULTS_DIR / "per_protein_spearman.csv")
kaggle_ranked_df = pd.read_csv(RESULTS_DIR / "kaggle_test_ranked_variants.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Benchmark Cohort", "2,766 Mutations", "36 Distinct UniProt Proteins")
c2.metric("GroupKFold Spearman ρ", "0.3909", "Kaggle Rank-Average Ensemble")
c3.metric("ΔΔG Cohort Mean ρ", "0.4733", "23 Held-Out Proteins")
c4.metric("Kaggle Test Submission", "2,413 Variants", "pH 8.0 Wildtype AF2")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Benchmark & Leakage Analysis",
        "🔬 Interactive Single-Mutation Scorer",
        "🏆 Kaggle Novozymes 2,413 Test Explorer",
        "🧬 36-Protein Cohort Table",
    ]
)

with tab1:
    st.subheader("Protein-Held-Out GroupKFold vs Random Split Leakage")
    st.dataframe(metrics_df, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(FIG_DIR / "spearman_benchmark.png"), caption="Spearman Rank Correlation Benchmark")
    with col_b:
        st.image(str(FIG_DIR / "plddt_vs_stability.png"), caption="AlphaFold pLDDT & 3D Packing vs Stability")
    st.image(str(FIG_DIR / "feature_importance.png"), caption="Permutation Feature Importance & Per-Protein ρ")

with tab2:
    st.subheader("Live Mutation Scorer on the 221-aa Kaggle Novozymes Wildtype Enzyme")
    profile = parse_alphafold_structure(DATA_DIR / "structures" / "wildtype_structure_prediction_af2.pdb")
    col1, col2 = st.columns([1, 1])
    with col1:
        pos = st.slider("Residue Position (1 to 221)", min_value=1, max_value=len(NOVOZYMES_WT_SEQUENCE), value=16)
        wt_aa = NOVOZYMES_WT_SEQUENCE[pos - 1]
        st.write(f"**Wildtype Amino Acid at Position {pos}:** `{wt_aa}`")
        mut_options = [aa for aa in sorted(KYTE_DOOLITTLE.keys()) if aa != wt_aa]
        mut_aa = st.selectbox("Mutant Amino Acid", mut_options, index=0)

    feats = compute_mutation_features(
        wt_aa=wt_aa,
        mut_aa=mut_aa,
        seq_pos_1based=pos,
        seq_len=len(NOVOZYMES_WT_SEQUENCE),
        af_res_num=pos,
        profile=profile,
    )
    with col2:
        st.markdown(f"### Mutation `{wt_aa}{pos}{mut_aa}` 3D Structural Profile")
        st.write(f"- **AlphaFold pLDDT Confidence:** `{feats['af_plddt']:.1f}` (Window: `{feats['af_window_plddt']:.1f}`)")
        st.write(f"- **3D 10 Å Cα Packing Neighbors:** `{int(feats['ca_packing_10a'])}` (15 Å: `{int(feats['ca_packing_15a'])}`)")
        st.write(f"- **BLOSUM62 Log-Odds Score:** `{int(feats['blosum62'])}`")
        st.write(f"- **ΔHydrophobicity (Kyte-Doolittle):** `{feats['delta_hydrophobicity']:+.2f}`")
        st.write(f"- **ΔSide-Chain Volume (Å³):** `{feats['delta_volume']:+.1f}`")
        st.json(feats)

with tab3:
    st.subheader("Official Kaggle Novozymes Test Set Ranked Predictions (2,413 Variants)")
    st.dataframe(kaggle_ranked_df.sort_values("tm", ascending=False), use_container_width=True)

with tab4:
    st.subheader("Per-Protein Held-Out Spearman Performance Across 36 Enzymes")
    st.dataframe(per_prot_df.sort_values("n_mutations", ascending=False), use_container_width=True)
