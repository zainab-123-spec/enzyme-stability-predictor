# Study Guide & Interview Prep: Kaggle Novozymes Enzyme Stability Benchmark

This guide explains every biological concept, feature, and machine learning design choice in **simple, beginner-friendly English** so you can confidently present this project on Kaggle, GitHub, or in a **Google Student Researcher / Bioinformatics Intern** interview.

---

## 1. The Big Picture in Plain English

### What problem does this project solve?
Enzymes are biological catalysts (proteins that speed up chemical reactions in medicine, laundry detergents, biofuel production, and recycling plastics). However, most natural enzymes unfold (**denature**) when heated.
- **Melting Temperature ($T_m$)**: The temperature (in °C) at which 50% of the enzyme is unfolded. A positive shift ($\Delta T_m > 0$) means the mutated enzyme is **more heat-stable**.
- **Folding Free Energy Change ($\Delta\Delta G$)**: The change in thermodynamic stability (in kcal/mol) caused by a single amino-acid mutation. In the FireProtDB/ProTherm convention used here, $\Delta\Delta G > 0$ means **stabilizing** and $\Delta\Delta G < 0$ means **destabilizing**.

### Why did Kaggle host the Novozymes Enzyme Stability Prediction competition?
Novozymes (the world's largest industrial enzyme company) challenged data scientists to predict the thermostability ranking of **2,413 single-amino-acid variants** of a 221-amino-acid industrial enzyme at $\text{pH } 8.0$, given its **AlphaFold2 3D structure** (`wildtype_structure_prediction_af2.pdb`).

### Why is the official metric **Spearman's Rank Correlation ($\rho$)** instead of RMSE?
When wet-lab biologists engineer an enzyme, they don't care whether a computer predicts $T_m = 64.2\,^\circ\text{C}$ vs $65.8\,^\circ\text{C}$—they care **which 20 mutations out of 2,413 sit at the very top of the list** so they can synthesize those 20 in the lab. Spearman's $\rho \in [-1, +1]$ measures how well your predicted ranking order matches the true experimental ranking order.

---

## 2. The 3 Core Insights Every Interviewer Will Love

1. **Why `-pLDDT` (Negative AlphaFold Confidence) Predicts Mutation Stability**:
   - In an AlphaFold PDB file, the `B-factor` column (`columns 61–66` of each `ATOM` line) stores **`pLDDT`** (predicted Local Distance Difference Test, scored `0–100`).
   - High `pLDDT` (`92–100`) marks rigid, tightly packed hydrophobic core residues that evolution has already perfected. Changing an amino acid in that rigid core almost always creates a steric clash or a cavity that **destabilizes** the protein (mean within-protein stability rank `0.49`, only `23%` stabilizing).
   - Low `pLDDT` (`< 75`) marks flexible surface loops. Mutating a flexible surface loop rarely breaks the core and frequently improves stability (mean within-protein stability rank `0.79`, `42%` stabilizing).
   - Therefore, **`-af_plddt` alone achieves $\rho = 0.2142$ globally ($\bar{\rho} = 0.2407$ within proteins)**—more than doubling `BLOSUM62` alone ($\rho = 0.1349$)!

2. **Why 3D $10\text{ \AA}$ $C_\alpha$ Packing Density (`ca_packing_10a`) Is Even Stronger**:
   - For every residue $i$, we parse its 3D $(x, y, z)$ coordinates from the AlphaFold PDB file and count how many other $C_\alpha$ atoms lie within a $10\text{ \AA}$ sphere (`np.linalg.norm(coords - coords[i], axis=1) <= 10.0`).
   - Surface residues have $\le 12$ neighbors; dense hydrophobic core residues have $\ge 23$ neighbors.
   - `-ca_packing_10a` achieves **$\rho = 0.2733$ globally ($\bar{\rho} = 0.3317$ on the 23-protein $\Delta\Delta G$ cohort)** without any machine learning training at all!

3. **Why We Use Protein-Held-Out `GroupKFold` + Rank-Averaging**:
   - If you split 2,766 mutations randomly (`Random KFold`), mutations from Staphylococcal nuclease (`1STN`, 526 mutations) appear in both the training set and the test set. Decision trees (`HistGradientBoosting`) memorize the protein's sequence position and jump to an artificial $\rho = 0.5133$.
   - When we hold out entire proteins (`GroupKFold` grouped by `uniprot_id`, **0 shared proteins**), `HistGradientBoosting` gets $\rho = 0.3786$, and our **Kaggle Rank-Average Ensemble** (`0.25 * Ridge + 0.25 * RandomForest + 0.30 * HistGradientBoosting + 0.20 * ZeroShotBiophysicalRank`) reaches **$\rho = 0.3909$ globally ($\bar{\rho} = 0.4733$ on the 23-protein $\Delta\Delta G$ cohort)**.

---

## 3. Code Tour (File by File)

- `enzstab/constants.py`: Defines the 36 benchmark proteins (23 $\Delta\Delta G$ proteins + 13 $\Delta T_m$ enzymes), the 221-aa Novozymes wildtype sequence, the symmetric $20 \times 20$ `BLOSUM62` matrix, Kyte-Doolittle hydrophobicity, Zimmerman side-chain volume, and Chou-Fasman secondary structure propensities.
- `enzstab/fetch.py`: Downloads experimental mutation datasets, queries live **AlphaFold DB API** (`https://alphafold.ebi.ac.uk/api/prediction/{uniprot}`), **UniProt REST API**, and **RCSB PDB REST API**, and caches the 36 AlphaFold v6 `.pdb` files plus the Novozymes test files.
- `enzstab/features.py`: Parses 3D $C_\alpha$ coordinates and `pLDDT` from `.pdb` files and computes the 19 sequence + 3D structural features.
- `enzstab/models.py`: Runs zero-shot baselines, 5-fold `GroupKFold` vs 5-fold `Random KFold`, permutation feature importance, and generates the 2,413-row Kaggle `submission.csv`.
- `enzstab/plots.py`: Produces the 3 publication figures in `docs/figures/`.

---

## 4. 15 Interview Questions & Answers

1. **Q: Why did you choose the Kaggle Novozymes Enzyme Stability competition for your bioinformatics portfolio?**  
   **A:** Because predicting how single-point mutations affect protein thermostability ($\Delta T_m$ and $\Delta\Delta G$) is a foundational problem in protein engineering, and the Novozymes competition is famous for showing that interpretable 3D AlphaFold features (`pLDDT` and $10\text{ \AA}$ $C_\alpha$ packing) combined with Spearman rank-ensembling beat complex black-box models that overfit to training proteins.

2. **Q: How is AlphaFold `pLDDT` stored inside a `.pdb` file, and how did you parse it?**  
   **A:** In AlphaFold PDB files, per-residue `pLDDT` (`0–100`) is stored in the temperature factor (`B-factor`) column (characters `61–66` of each `ATOM` record). In `parse_alphafold_structure()`, I filter for `ATOM` lines where the atom name (`columns 13–16`) is `CA`, extract $(x, y, z)$ from columns `31–54`, and read `float(line[60:66])` as `pLDDT`.

3. **Q: Why does a higher AlphaFold `pLDDT` at the mutated site correlate with lower thermostability after mutation?**  
   **A:** `pLDDT` measures local structural confidence and rigidity. Residues with `pLDDT >= 96` sit in tightly packed, well-ordered hydrophobic cores where side-chain packing is already near optimal. Substituting a residue there usually introduces steric clashes or leaves an energetic cavity, lowering stability.

4. **Q: How did you compute 3D $10\text{ \AA}$ $C_\alpha$ packing density (`ca_packing_10a`)?**  
   **A:** After extracting the $N \times 3$ array of $C_\alpha$ coordinates from the AlphaFold PDB, I compute the $N \times N$ pairwise Euclidean distance matrix `np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)`, set the diagonal to $\infty$, and count how many neighbors lie within $\le 10.0\text{ \AA}$ for each residue.

5. **Q: Why did you normalize experimental $\Delta\Delta G$ and $\Delta T_m$ values to within-protein percentile ranks (`within_protein_rank`) during training?**  
   **A:** Because $\Delta\Delta G$ is measured in $\text{kcal/mol}$ (across 23 proteins) and $\Delta T_m$ is measured in $^\circ\text{C}$ (across 13 enzymes), and different experimental assays have different dynamic ranges. Converting each protein's mutations to percentile ranks in $(0, 1]$ via `scipy.stats.rankdata` puts all 36 proteins on a common scale directly aligned with Kaggle's Spearman $\rho$ objective.

6. **Q: How did you align PDB construct sequences to full-length UniProt AlphaFold structures?**  
   **A:** Experimental crystal structures in ProTherm/FireProt often cleave off N-terminal signal peptides (for example, `1STN` starts at residue 88 of full-length *S. aureus* nuclease `P00644`). My `find_sequence_offset()` function locates the exact substring offset of the construct sequence inside the full-length AlphaFold sequence so every mutation maps to its exact 3D residue (`100%` wild-type amino-acid verification across all 2,766 mutations).

7. **Q: What happens when you evaluate with Random `KFold` vs Protein-Held-Out `GroupKFold`?**  
   **A:** Under 5-fold Random `KFold`, all 36 proteins appear in both train and validation splits, allowing `HistGradientBoosting` to memorize protein-level coordinates and inflate Spearman $\rho$ from `0.3786` to `0.5133`. Under 5-fold `GroupKFold` (`groups=uniprot_id`), zero proteins are shared between train and validation, giving an honest measure of generalization to a new enzyme.

8. **Q: Why does a Spearman Rank-Average Ensemble beat every individual model on held-out proteins?**  
   **A:** Linear models (`Ridge`), bagging trees (`Random Forest`), boosting trees (`HistGradientBoosting`), and the zero-shot biophysical rule (`BLOSUM62 - pLDDT - 3D packing - core clash`) make complementary errors. Averaging their percentile ranks reduces variance and achieves `0.3909` Global Spearman $\rho$ (`0.4733` mean $\rho$ across the 23 $\Delta\Delta G$ proteins).

9. **Q: What is the `core_hydrophobic_clash` interaction feature?**  
   **A:** It multiplies normalized $10\text{ \AA}$ packing density $(\text{ca\_packing\_10a} / 20)$ by the absolute change in Kyte-Doolittle hydropathy $|\Delta H|$. Changing a hydrophobic residue to a polar residue on the exposed surface is harmless, but doing the same inside a densely packed core is strongly destabilizing.

10. **Q: Why did you include `is_proline_mutation` and `is_glycine_mutation` as explicit binary features?**  
    **A:** Proline has a cyclic side chain bonded to the backbone nitrogen, restricting the backbone $\phi$ angle and breaking $\alpha$-helices (`helix breaker`). Glycine has no side chain ($\text{-H}$), giving the unfolded polypeptide chain high conformational entropy. Mutating to or from either residue disproportionately shifts folding free energy.

11. **Q: How are single-amino-acid deletions handled in the 2,413 Kaggle Novozymes test variants?**  
    **A:** In the 2,413 test sequences, most variants have length `221` (single substitutions), while a small subset has length `220` (single-residue deletions). Removing an amino acid from the backbone usually disrupts local secondary structure, so deletions receive a `-6` BLOSUM penalty and are scaled to the bottom stability decile in `generate_kaggle_submission()`.

12. **Q: Which features had the highest permutation importance in `HistGradientBoosting`?**  
    **A:** The top features by permutation importance ($\Delta R^2$) are `af_plddt` (`0.100`), `ca_packing_10a` (`0.091`), `delta_sheet_prop` (`0.090`), `ca_packing_15a` (`0.067`), `rel_seq_pos` (`0.064`), and `delta_hydrophobicity` (`0.052`), confirming that 3D AlphaFold structure and physicochemical deltas drive the predictions.

13. **Q: Why is `ddG_cohort_spearman` (`0.4733`) higher than `dtm_cohort_spearman` (`0.1371`–`0.1924`)?**  
    **A:** Direct $\Delta\Delta G$ measurements come from reversible two-state equilibrium unfolding experiments on well-characterized monomeric domains (`1STN`, `1BNI`, `1LZ1`), whereas apparent $\Delta T_m$ assays across industrial enzymes often involve irreversible thermal aggregation and kinetic effects across diverse pH buffers.

14. **Q: How can a student extend this pipeline with Protein Language Models (PLMs)?**  
    **A:** You can add the zero-shot log-likelihood ratio $\log P(\text{Mut}_i \mid \text{context}) - \log P(\text{WT}_i \mid \text{context})$ from `ESM-2` (`esm2_t33_650M_UR50D`) as an additional rank component alongside `Ridge`, `HistGradientBoosting`, and `AlphaFold pLDDT`.

15. **Q: How do you upload this solution directly to Kaggle?**  
    **A:** Either upload `results/submission.csv` (`2,413` rows with `seq_id,tm`) directly to the Kaggle Novozymes competition page, or upload `kaggle_enzyme_thermostability_solution.ipynb` under the **Code -> New Notebook -> Import Notebook** tab on Kaggle.
