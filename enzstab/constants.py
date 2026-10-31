"""Curated 36-protein benchmark cohort, Novozymes wildtype sequence, and biophysical scales."""

from __future__ import annotations

# Official 221-aa wildtype enzyme sequence from Kaggle Novozymes Enzyme Stability Prediction
NOVOZYMES_WT_SEQUENCE = (
    "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQ"
    "RVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGT"
    "NAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKAL"
    "GSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
)

# 36 distinct UniProt proteins/enzymes with verified AlphaFold DB v6 3D structures
# (23 ProTherm/FireProt ddG proteins + 13 FireProt/ProTherm dTm enzymes)
BENCHMARK_PROTEINS: list[dict[str, str]] = [
    # --- ddG cohort (kcal/mol; >0 stabilizing, <0 destabilizing) ---
    {"pdb_id": "1STN", "uniprot_id": "P00644", "name": "Staphylococcal nuclease", "organism": "Staphylococcus aureus", "assay": "ddG"},
    {"pdb_id": "1BNI", "uniprot_id": "P00648", "name": "Barnase ribonuclease", "organism": "Bacillus amyloliquefaciens", "assay": "ddG"},
    {"pdb_id": "1LZ1", "uniprot_id": "P61626", "name": "Lysozyme C", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "1RX4", "uniprot_id": "P0ABQ4", "name": "Dihydrofolate reductase", "organism": "Escherichia coli", "assay": "ddG"},
    {"pdb_id": "2RN2", "uniprot_id": "P0A7Y4", "name": "Ribonuclease H", "organism": "Escherichia coli", "assay": "ddG"},
    {"pdb_id": "1BVC", "uniprot_id": "P02185", "name": "Myoglobin", "organism": "Physeter catodon", "assay": "ddG"},
    {"pdb_id": "1BU4", "uniprot_id": "P00651", "name": "Ribonuclease T1", "organism": "Aspergillus oryzae", "assay": "ddG"},
    {"pdb_id": "1SHF", "uniprot_id": "P06241", "name": "Tyrosine-protein kinase Fyn SH3", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "1RGG", "uniprot_id": "P05798", "name": "Guanyl-specific ribonuclease Sa", "organism": "Streptomyces aureofaciens", "assay": "ddG"},
    {"pdb_id": "1AJ3", "uniprot_id": "P07751", "name": "Spectrin alpha chain SH3", "organism": "Gallus gallus", "assay": "ddG"},
    {"pdb_id": "1BPI", "uniprot_id": "P00974", "name": "Pancreatic trypsin inhibitor (BPTI)", "organism": "Bos taurus", "assay": "ddG"},
    {"pdb_id": "4GWT", "uniprot_id": "Q13526", "name": "Peptidyl-prolyl isomerase Pin1", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "4LYZ", "uniprot_id": "P00698", "name": "Hen egg-white lysozyme C", "organism": "Gallus gallus", "assay": "ddG"},
    {"pdb_id": "4HXJ", "uniprot_id": "P12931", "name": "Proto-oncogene tyrosine kinase Src", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "1RTB", "uniprot_id": "P61823", "name": "Pancreatic ribonuclease A", "organism": "Bos taurus", "assay": "ddG"},
    {"pdb_id": "1FKJ", "uniprot_id": "P62942", "name": "Peptidyl-prolyl isomerase FKBP1A", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "1CSP", "uniprot_id": "P32081", "name": "Cold shock protein CspB", "organism": "Bacillus subtilis", "assay": "ddG"},
    {"pdb_id": "2HBB", "uniprot_id": "P02417", "name": "30S ribosomal protein S6", "organism": "Thermus thermophilus", "assay": "ddG"},
    {"pdb_id": "2TRX", "uniprot_id": "P0AA25", "name": "Thioredoxin-1", "organism": "Escherichia coli", "assay": "ddG"},
    {"pdb_id": "1UZC", "uniprot_id": "O75400", "name": "Pre-mRNA-processing factor 40A", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "1FTG", "uniprot_id": "P0A3E0", "name": "Flavodoxin", "organism": "Desulfovibrio vulgaris", "assay": "ddG"},
    {"pdb_id": "1TUP", "uniprot_id": "P04637", "name": "Cellular tumor antigen p53", "organism": "Homo sapiens", "assay": "ddG"},
    {"pdb_id": "1UBQ", "uniprot_id": "P0CG48", "name": "Ubiquitin", "organism": "Homo sapiens", "assay": "ddG"},
    # --- dTm cohort (melting temperature shift in Celsius; >0 stabilizing, <0 destabilizing) ---
    {"pdb_id": "6G4B", "uniprot_id": "A0A2W0F5X5", "name": "Thermophilic omega-transaminase", "organism": "Thermomicrobium roseum", "assay": "dTm"},
    {"pdb_id": "1PX0", "uniprot_id": "Q93D82", "name": "Haloalkane dehalogenase DhaA", "organism": "Rhodococcus rhodochrous", "assay": "dTm"},
    {"pdb_id": "3WP4", "uniprot_id": "Q9UV68", "name": "Endo-1,4-beta-xylanase", "organism": "Aspergillus niger", "assay": "dTm"},
    {"pdb_id": "2O9P", "uniprot_id": "P22505", "name": "Endoglucanase Cel5A", "organism": "Bacillus agaradhaerens", "assay": "dTm"},
    {"pdb_id": "3VUB", "uniprot_id": "P62554", "name": "Thermostable beta-glucosidase", "organism": "Thermotoga maritima", "assay": "dTm"},
    {"pdb_id": "3TGL", "uniprot_id": "P19515", "name": "Triacylglycerol lipase", "organism": "Rhizomucor miehei", "assay": "dTm"},
    {"pdb_id": "6BQG", "uniprot_id": "P28335", "name": "5-HT2C receptor construct", "organism": "Homo sapiens", "assay": "dTm"},
    {"pdb_id": "4E5K", "uniprot_id": "O69054", "name": "Limonene-1,2-epoxide hydrolase", "organism": "Rhodococcus erythropolis", "assay": "dTm"},
    {"pdb_id": "4ZLU", "uniprot_id": "B3VI55", "name": "Cutinase / PETase homolog", "organism": "Streptomyces sp.", "assay": "dTm"},
    {"pdb_id": "3D2A", "uniprot_id": "P37957", "name": "Lipase EstA (BsubL)", "organism": "Bacillus subtilis", "assay": "dTm"},
    {"pdb_id": "1CF3", "uniprot_id": "P13006", "name": "Glucose oxidase", "organism": "Aspergillus niger", "assay": "dTm"},
    {"pdb_id": "1CQW", "uniprot_id": "P59336", "name": "Haloalkane dehalogenase LinB", "organism": "Sphingobium japonicum", "assay": "dTm"},
    {"pdb_id": "1M21", "uniprot_id": "Q8RJN5", "name": "Histidine acid phytase", "organism": "Yersinia intermedia", "assay": "dTm"},
]

AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Kyte-Doolittle hydropathy index (J. Mol. Biol. 1982)
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Zimmerman side-chain volume in Angstrom^3 (J. Theor. Biol. 1968)
RESIDUE_VOLUME: dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Chou-Fasman alpha-helix propensity (Biochemistry 1974)
HELIX_PROPENSITY: dict[str, float] = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

# Chou-Fasman beta-sheet propensity (Biochemistry 1974)
SHEET_PROPENSITY: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Approximate net side-chain charge at physiological pH 7.4
RESIDUE_CHARGE: dict[str, float] = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0,
    "Q": 0.0, "E": -1.0, "G": 0.0, "H": 0.1, "I": 0.0,
    "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}

# Standard symmetric BLOSUM62 substitution matrix
_BLOSUM62_ROWS: dict[str, list[int]] = {
    "A": [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    "R": [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    "N": [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    "D": [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    "C": [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "Q": [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    "E": [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    "G": [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    "H": [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    "K": [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    "M": [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    "S": [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    "T": [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    "V": [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],
}
_AA_ORDER = list(_BLOSUM62_ROWS.keys())
BLOSUM62: dict[tuple[str, str], int] = {
    (a, b): _BLOSUM62_ROWS[a][j]
    for a in _AA_ORDER
    for j, b in enumerate(_AA_ORDER)
}


def get_blosum62(wt: str, mut: str) -> int:
    """Return symmetric BLOSUM62 substitution log-odds score (or -6 for deletions/indels)."""
    return BLOSUM62.get((str(wt).upper(), str(mut).upper()), -6)
