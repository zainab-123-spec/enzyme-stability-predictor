"""Live data fetcher for ProTherm/FireProt mutations, AlphaFold DB, UniProt, RCSB PDB, and Kaggle Novozymes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from enzstab.constants import BENCHMARK_PROTEINS

JIN_BASE_URL = "https://raw.githubusercontent.com/JinyuanSun/mutation-stability-data/main"
NOVO_BASE_URL = (
    "https://raw.githubusercontent.com/martin-he543/"
    "novozymes-enzyme-stability-prediction-kaggle-2022/main/data"
)
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction"
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb"
RCSB_API_URL = "https://data.rcsb.org/rest/v1/core/entry"


def fetch_protein_metadata(
    uniprot_id: str,
    pdb_id: str,
    cache_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch live metadata from AlphaFold DB, UniProt REST, and RCSB PDB REST APIs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{pdb_id}_{uniprot_id}_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    sess = session or requests.Session()
    meta: dict[str, Any] = {
        "pdb_id": pdb_id,
        "uniprot_id": uniprot_id,
        "af_model_id": f"AF-{uniprot_id}-F1",
        "af_pdb_url": f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb",
        "af_GlobalMetricValue": None,
        "uniprot_length": None,
        "uniprot_organism": None,
        "rcsb_resolution": None,
        "rcsb_method": None,
    }

    # 1. AlphaFold DB API
    af_resp = sess.get(f"{ALPHAFOLD_API_URL}/{uniprot_id}", timeout=15)
    if af_resp.status_code == 200 and af_resp.json():
        entry = af_resp.json()[0]
        meta["af_model_id"] = entry.get("modelEntityId", meta["af_model_id"])
        meta["af_pdb_url"] = entry.get("pdbUrl", meta["af_pdb_url"])
        meta["af_GlobalMetricValue"] = entry.get("globalMetricValue")

    # 2. UniProt REST API
    up_resp = sess.get(f"{UNIPROT_API_URL}/{uniprot_id}.json", timeout=15)
    if up_resp.status_code == 200:
        up_data = up_resp.json()
        meta["uniprot_length"] = up_data.get("sequence", {}).get("length")
        meta["uniprot_organism"] = up_data.get("organism", {}).get("scientificName")

    # 3. RCSB PDB REST API
    rcsb_resp = sess.get(f"{RCSB_API_URL}/{pdb_id.lower()}", timeout=15)
    if rcsb_resp.status_code == 200:
        rcsb_data = rcsb_resp.json()
        res_combined = rcsb_data.get("rcsb_entry_info", {}).get("resolution_combined")
        if res_combined:
            meta["rcsb_resolution"] = float(res_combined[0])
        exptl = rcsb_data.get("exptl", [])
        if exptl:
            meta["rcsb_method"] = exptl[0].get("method")

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def download_alphafold_pdb(
    uniprot_id: str,
    pdb_url: str,
    struct_dir: Path,
    session: requests.Session | None = None,
) -> Path:
    """Download and cache an AlphaFold DB PDB coordinate file."""
    struct_dir.mkdir(parents=True, exist_ok=True)
    out_path = struct_dir / f"AF-{uniprot_id}-F1.pdb"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path

    sess = session or requests.Session()
    resp = sess.get(pdb_url, timeout=20)
    resp.raise_for_status()
    out_path.write_text(resp.text, encoding="utf-8")
    return out_path


def fetch_all_raw_datasets(data_dir: Path) -> dict[str, Path]:
    """Download raw mutation stability tables, AlphaFold 3D structures, and Kaggle Novozymes test files."""
    raw_dir = data_dir / "raw"
    struct_dir = data_dir / "structures"
    meta_dir = raw_dir / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()

    # 1. Experimental ddG and dTm datasets (ProThermDB / FireProtDB curated benchmark)
    ddg_path = raw_dir / "protherm_ddg_raw.csv"
    if not ddg_path.exists():
        r = sess.get(f"{JIN_BASE_URL}/train.csv", timeout=20)
        r.raise_for_status()
        ddg_path.write_text(r.text, encoding="utf-8")

    dtm_path = raw_dir / "fireprot_dtm_raw.csv"
    if not dtm_path.exists():
        r = sess.get(f"{JIN_BASE_URL}/tm.csv", timeout=20)
        r.raise_for_status()
        dtm_path.write_text(r.text, encoding="utf-8")

    # 2. Official Kaggle Novozymes test.csv, sample_submission.csv, and wildtype AF2 PDB
    novo_test_path = raw_dir / "kaggle_novozymes_test.csv"
    if not novo_test_path.exists():
        r = sess.get(f"{NOVO_BASE_URL}/test.csv", timeout=20)
        r.raise_for_status()
        novo_test_path.write_text(r.text, encoding="utf-8")

    novo_sub_path = raw_dir / "kaggle_sample_submission.csv"
    if not novo_sub_path.exists():
        r = sess.get(f"{NOVO_BASE_URL}/sample_submission.csv", timeout=20)
        r.raise_for_status()
        novo_sub_path.write_text(r.text, encoding="utf-8")

    novo_pdb_path = struct_dir / "wildtype_structure_prediction_af2.pdb"
    if not novo_pdb_path.exists():
        r = sess.get(f"{NOVO_BASE_URL}/wildtype_structure_prediction_af2.pdb", timeout=20)
        r.raise_for_status()
        novo_pdb_path.write_text(r.text, encoding="utf-8")

    # 3. Fetch AlphaFold DB, UniProt, and RCSB PDB metadata + AF2 structures for all 36 benchmark proteins
    meta_rows = []
    for p in BENCHMARK_PROTEINS:
        meta = fetch_protein_metadata(p["uniprot_id"], p["pdb_id"], meta_dir, session=sess)
        meta["protein_name"] = p["name"]
        meta["organism"] = p["organism"]
        meta["assay"] = p["assay"]
        download_alphafold_pdb(p["uniprot_id"], meta["af_pdb_url"], struct_dir, session=sess)
        meta_rows.append(meta)

    meta_summary_path = raw_dir / "protein_cohort_metadata.csv"
    pd.DataFrame(meta_rows).to_csv(meta_summary_path, index=False)

    return {
        "ddg_raw": ddg_path,
        "dtm_raw": dtm_path,
        "novo_test": novo_test_path,
        "novo_sub": novo_sub_path,
        "novo_pdb": novo_pdb_path,
        "meta_summary": meta_summary_path,
    }
