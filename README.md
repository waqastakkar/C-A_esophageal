# C-A_esophageal

## Project overview

This repository implements a staged TCGA ESCA tumor-vs-normal modeling workflow aligned to the study design below:

1. **Stage I — Cohort curation and preprocessing**
2. **Stage II — Deep invariant sparse modeling**
3. **Stage III — Candidate driver prioritization and downstream validation**

Step 5 is now the first major Stage III implementation. It converts the Step 4 deep invariant sparse model outputs into a careful, computationally grounded **Candidate Driver Priority Score (CDPS)** framework that ranks genes using multiple evidence channels without claiming causal proof.

## Pipeline overview

1. **Step 1 – generic preprocessing scaffold**
2. **Step 2 – curated primary cohort construction**
3. **Step 3 – grouped nested baseline benchmarking**
4. **Step 4 – sparse invariant adversarial deep disease model**
5. **Step 5 – candidate driver prioritization with CDPS**
6. **Step 6 – next planned step: biological/statistical validation with raw counts, enrichment, networks, and clinicopathologic analyses**

## Figure Style Standard

All figures in this project must follow the shared publication rules below.

- **SVG is mandatory** for primary figure export.
- **All text must be bold**.
- **Roman/serif fonts only** with preferred order: Times New Roman → Times → STIXGeneral → DejaVu Serif.
- **Reuse `publication_style.py`** for plotting configuration and figure saving.
- **Use the shared Nature-style palette** from `publication_style.py` rather than local ad hoc colors.
- Keep figures manuscript-ready: compact legends, thicker axes, minimal chart junk, balanced whitespace, and editable SVG text where possible.
- These rules remain mandatory for **all future steps** in the project.

## Suggested project structure

```text
C-A_esophageal/
├── preprocess_esca.py
├── step2_build_cohort.py
├── step3_grouped_baselines.py
├── step4_sparse_invariant_model.py
├── step5_cdps_ranking.py
├── deep_model_utils.py
├── publication_style.py
├── README.md
└── outputs/
    ├── step1_preprocess/
    ├── step2_cohort/
    ├── step3_baselines/
    ├── step4_deep_model/
    └── step5_cdps/
```

## Dependency notes

Core dependencies:

- Python 3.10+
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- **PyTorch**

Optional dependencies:

- `xgboost`
- `lightgbm`
- `umap-learn`
- pathway GMT file inputs for lightweight pathway support in Step 5

If optional packages or artifacts are unavailable, the pipeline degrades gracefully and records the limitation in saved outputs and console logs.

---

## Step 1 – generic preprocessing scaffold

### Purpose

Provides reusable loading, harmonization, scaling, and exploratory export utilities for counts, metadata, and normalized matrices.

### Key inputs

- `TCGA_ESCA_STAR_Counts.csv`
- `TCGA_ESCA_Metadata.csv`
- `ESCA_vst_normalized_matrix.csv`

### Main logic

- Detects and harmonizes identifiers.
- Handles matrix orientation.
- Imputes missing values and applies optional scaling.
- Can auto-detect a label column and export a plain train/test split.

### Outputs

- `TCGA_ESCA_preprocessed.csv`
- `TCGA_ESCA_X_train.csv`
- `TCGA_ESCA_X_test.csv`
- `TCGA_ESCA_y_train.csv`
- `TCGA_ESCA_y_test.csv`
- QC diagnostics

### Example command

```bash
python preprocess_esca.py \
  --counts TCGA_ESCA_STAR_Counts.csv \
  --metadata TCGA_ESCA_Metadata.csv \
  --normalized ESCA_vst_normalized_matrix.csv \
  --output-dir outputs/step1_preprocess
```

### Leakage / study-design note

Step 1 is a **generic scaffold only**. Its auto-detected labels and plain sample-level split are not the final training design for the study.

---

## Step 2 – curated primary cohort construction

### Purpose

Builds the study-grounded primary tumor-vs-normal cohort and the grouped outer folds required for leakage-safe evaluation.

### Key inputs

- `TCGA_ESCA_STAR_Counts.csv`
- `ESCA_vst_normalized_matrix.csv`
- `TCGA_ESCA_Metadata.csv`

### Main logic

- Harmonizes TCGA identifiers.
- Restricts to supported primary tumor and solid tissue normal samples.
- Creates explicit `disease_label` values.
- Curates metadata for downstream confounder/environment use.
- Selects representative primary samples per patient.
- Creates grouped outer folds.
- Exports aligned count and normalized matrices.

### Outputs

- `master_samples_primary.csv`
- `normalized_primary_matrix.csv`
- `counts_primary_matrix.csv`
- `grouped_outer_folds.csv`
- cohort SVG QC figures

### Example command

```bash
python step2_build_cohort.py \
  --counts TCGA_ESCA_STAR_Counts.csv \
  --normalized ESCA_vst_normalized_matrix.csv \
  --metadata TCGA_ESCA_Metadata.csv \
  --output-dir outputs/step2_cohort
```

### Leakage / study-design note

Step 2 is the **true entry point** for all later model training and grouped evaluation. Grouped folds prevent patient overlap between train and outer-test partitions.

---

## Step 3 – grouped nested baseline benchmarking

### Purpose

Benchmarks grouped nested baseline classifiers on the curated Step 2 cohort before deep invariant sparse modeling.

### Key inputs

- `outputs/step2_cohort/master_samples_primary.csv`
- `outputs/step2_cohort/normalized_primary_matrix.csv`
- `outputs/step2_cohort/grouped_outer_folds.csv`

### Main logic

- Uses explicit `disease_label` values.
- Aligns cohort, normalized matrix, and grouped folds by `sample_id`.
- Uses grouped outer folds and grouped inner CV only.
- Performs variable-gene selection and scaling on training data only.
- Trains and evaluates logistic regression L1, elastic net, random forest, gradient boosting fallback, and optional MLP / boosted baselines.
- Produces pooled out-of-fold predictions, subgroup summaries, calibration data, and manuscript-grade SVG figures.

### Outputs

- `baseline_fold_metrics.csv`
- `baseline_summary_metrics.csv`
- `baseline_oof_predictions.csv`
- `baseline_subgroup_metrics.csv`
- `model_selection_log.csv`
- `baseline_feature_space_summary.csv`
- `calibration_data.csv`
- `run_config.json`
- Step 3 SVG figures

### Example command

```bash
python step3_grouped_baselines.py \
  --cohort outputs/step2_cohort/master_samples_primary.csv \
  --normalized outputs/step2_cohort/normalized_primary_matrix.csv \
  --folds outputs/step2_cohort/grouped_outer_folds.csv \
  --output-dir outputs/step3_baselines \
  --random-seeds 42,52,62,72,82 \
  --inner-folds 3 \
  --run-mlp \
  --run-xgboost
```

### Leakage / study-design note

Step 3 forbids sample-level random splits, patient overlap across partitions, outer-test-informed preprocessing, and metadata leakage into the default expression-only predictive input.

---

## Step 4 – sparse invariant adversarial deep disease model

### Purpose

Implements the main Stage II disease model for this study: a **Sparse Invariant Adversarial Autoencoder-Classifier** for TCGA ESCA tumor-vs-normal classification.

### Key inputs

- `outputs/step2_cohort/master_samples_primary.csv`
- `outputs/step2_cohort/normalized_primary_matrix.csv`
- `outputs/step2_cohort/grouped_outer_folds.csv`
- `outputs/step3_baselines/baseline_summary_metrics.csv`
- `outputs/step3_baselines/baseline_oof_predictions.csv`

### Main logic

- Uses **normalized expression only** as the predictive input.
- Uses metadata-derived confounders and environments for adversarial and invariance objectives rather than appending them as default predictive covariates.
- Aligns all Step 2/3 inputs by `sample_id` and validates grouped leakage safety.
- Selects variable genes using **training data only inside each outer fold**.
- Fits imputers/scalers using **training statistics only**.
- Uses grouped inner validation for modest hyperparameter selection and early stopping.
- Trains a compact PyTorch model with classification, reconstruction, sparsity, adversarial, and environment-invariance objectives.
- Exports out-of-fold predictions, fold metrics, training histories, latent embeddings, gate weights, environment summaries, and baseline-vs-deep comparisons.
- Saves manuscript-grade SVG figures through `publication_style.py`.

### Outputs

- `deep_oof_predictions.csv`
- `gate_weights.csv`
- `latent_embeddings.csv`
- `deep_summary_metrics.csv`
- `deep_fold_metrics.csv`
- `deep_training_history.csv`
- `environment_performance.csv`
- `invariance_summary.csv`
- optional fold checkpoints in `models/`
- Step 4 SVG figures

### Example command

```bash
python step4_sparse_invariant_model.py \
  --cohort outputs/step2_cohort/master_samples_primary.csv \
  --normalized outputs/step2_cohort/normalized_primary_matrix.csv \
  --folds outputs/step2_cohort/grouped_outer_folds.csv \
  --baseline-summary outputs/step3_baselines/baseline_summary_metrics.csv \
  --baseline-oof outputs/step3_baselines/baseline_oof_predictions.csv \
  --output-dir outputs/step4_deep_model \
  --random-seeds 42,52,62,72,82 \
  --save-fold-models \
  --save-latent-embeddings
```

### Leakage / study-design note

Step 4 preserves fold separation and does **not** refit using outer-test data. Attribution or interpretability work in later steps must operate on the already trained fold models rather than retraining on the full cohort.

---

## Step 5 – candidate driver prioritization with CDPS

### Purpose

Implements the first major Stage III module: computational prioritization of **candidate driver genes** using evidence from the Step 4 deep invariant sparse model.

### What CDPS is

The **Candidate Driver Priority Score (CDPS)** is a configurable, multi-component prioritization score that integrates:

1. **Predictive importance** from deep-model attribution.
2. **Model-derived sparsity support** from gate weights.
3. **Selection stability** across folds, seeds, and bootstrap resamples.
4. **Environment consistency / invariance** across metadata-defined subgroups.
5. **In silico perturbation impact** within the learned disease model.

CDPS is a **computational prioritization score**, not a causal proof score. The Step 5 outputs should be interpreted as rankings of candidate driver-associated genes or putative disease-promoting genes with stable disease-linked effects in the learned model.

### Why the score is multi-component

Single-method rankings can be dominated by one unstable run or one modeling artifact. Step 5 therefore combines complementary signals so that highly ranked genes are supported by both predictive evidence and reproducibility-oriented evidence.

### Key inputs

Required:

- `outputs/step2_cohort/master_samples_primary.csv`
- `outputs/step2_cohort/normalized_primary_matrix.csv`
- `outputs/step2_cohort/grouped_outer_folds.csv`
- `outputs/step4_deep_model/deep_oof_predictions.csv`
- `outputs/step4_deep_model/gate_weights.csv`
- `outputs/step4_deep_model/deep_summary_metrics.csv`
- `outputs/step4_deep_model/models/` for fold checkpoints when attribution/perturbation are desired

Optional:

- `outputs/step4_deep_model/latent_embeddings.csv`
- `outputs/step4_deep_model/deep_training_history.csv`
- gene annotation table
- pathway GMT file for lightweight pathway-level support summaries

### Main logic

#### 1. Attribution

Step 5 computes per-gene importance using:

- **Integrated Gradients**
- **Gradient × Input**
- **Gate-weight evidence** from Step 4

If full attribution across all fold models is too expensive, the script uses a **reproducible class-balanced sample cap per fold/seed** and records the analysis settings in `run_config.json`.

#### 2. Selection stability

Step 5 quantifies reproducibility across runs using:

- top-25 and top-100 selection frequency
- mean rank and rank variability
- coefficient-of-variation style spread
- bootstrap resampling of run-level evidence to obtain CDPS uncertainty intervals

#### 3. Environment invariance

Step 5 evaluates whether gene-level effects remain present across available metadata-defined environments such as:

- sex
- smoking
- histology
- stage
- country/region

The current implementation summarizes environment coverage, mean subgroup effect, subgroup variability, and worst-versus-best subgroup gaps. This rewards genes that remain active across environments instead of being driven by a single stratum.

#### 4. In silico perturbation

For top-ranked genes, Step 5 performs counterfactual perturbation within the learned Step 4 disease model:

- tumor samples are shifted toward a normal-like reference value
- normal samples are shifted toward a tumor-like reference value
- predicted disease probability is recomputed
- latent movement is summarized when latent output is available from the loaded model

This is reported as **counterfactual impact within the learned disease model**, not as causal intervention proof.

#### 5. Composite ranking

By default, Step 5 combines normalized components as:

```text
CDPS =
  0.30 * attribution_score
+ 0.20 * gate_score
+ 0.20 * stability_score
+ 0.15 * invariance_score
+ 0.15 * perturbation_score
```

The weights are CLI-configurable.

#### 6. Pathway-level support summary

If a pathway GMT file is provided, Step 5 builds a lightweight ranking-oriented pathway table containing:

- pathway gene count among ranked genes
- top-25 and top-100 hit counts
- mean CDPS
- mean attribution support
- mean perturbation support

This is **not** a formal enrichment test; full biological validation is deferred to Step 6.

### Outputs

Core tables:

- `gene_attribution_summary.csv`
- `samplewise_attributions.csv` when `--save-samplewise-attributions` is used
- `gate_importance_summary.csv`
- `gene_stability_summary.csv`
- `bootstrap_ranking_summary.csv`
- `gene_invariance_summary.csv`
- `environment_gene_effects.csv`
- `gene_perturbation_summary.csv`
- `perturbation_samplewise.csv`
- `ranked_genes_cdps.csv`
- `top25_genes_cdps.csv`
- `top100_genes_cdps.csv`
- `pathway_ranking_summary.csv` when pathway mode is enabled
- `latent_perturbation_summary.csv` when latent support is available
- `run_config.json`

Figures (all SVG, publication-style):

- `figure6_cdps_top_genes.svg`
- `figure6_stability_heatmap.svg`
- `figure6_attribution_vs_perturbation.svg`
- `figure6_cdps_components.svg`
- `figure6_environment_consistency.svg`
- `figure6_counterfactual_perturbation.svg`
- `figure6_pathway_summary.svg` when pathway mode is enabled
- `figure6_latent_shift.svg` when latent-shift summaries are available

### Example command

```bash
python step5_cdps_ranking.py \
  --cohort outputs/step2_cohort/master_samples_primary.csv \
  --normalized outputs/step2_cohort/normalized_primary_matrix.csv \
  --folds outputs/step2_cohort/grouped_outer_folds.csv \
  --deep-oof outputs/step4_deep_model/deep_oof_predictions.csv \
  --gate-weights outputs/step4_deep_model/gate_weights.csv \
  --latent-embeddings outputs/step4_deep_model/latent_embeddings.csv \
  --deep-metrics outputs/step4_deep_model/deep_summary_metrics.csv \
  --deep-model-dir outputs/step4_deep_model/models \
  --output-dir outputs/step5_cdps \
  --bootstrap-repeats 100 \
  --top-genes-for-perturbation 100 \
  --top-genes-for-detailed-report 25 \
  --save-samplewise-attributions \
  --run-pathway-summary
```

### Leakage / study-design note

Step 5 does **not** retrain or retune the Step 4 model. It operates fold-by-fold on already trained Step 4 artifacts and aggregates evidence afterward. Bootstrap resampling is applied to the evidence summaries rather than fitting new models on relabeled data.

### What Step 5 intentionally does **not** claim

Step 5 does **not** claim:

- proven biological causality
- that observational RNA-seq alone establishes true driver genes
- that counterfactual model perturbation is equivalent to real intervention

Use careful language such as:

- candidate driver genes
- computationally prioritized driver-associated genes
- genes with stable disease-linked effects
- counterfactual impact within the learned disease model

### What is intentionally deferred to Step 6

Step 6 will perform the heavier biological/statistical validation work, including:

- differential expression validation on raw counts
- formal enrichment testing
- pathway/network analysis
- survival or stage association testing
- clinicopathologic validation
- external validation where available

Those items are **not yet implemented in Step 5**.

---

## Next step: Step 6

The next planned module is **biological/statistical validation with raw counts and pathway/network/clinicopathologic analyses**. It will take the Step 5 ranked candidate lists and test their support using downstream validation analyses rather than extending the predictive model itself.


---

## Step 12 – reproducibility, release, and archival packaging

### Purpose

Step 12 creates the final reproducibility and release bundle for the study by validating previously generated artifacts, copying them into a deterministic package layout, generating a machine-readable manifest, computing checksums, capturing an environment snapshot, and optionally producing compressed archives for handoff or archival use.

### Key behavior

- Reads all Step-12 settings from `config.yaml` under the `script_12` section.
- Validates required package assets before packaging proceeds.
- Builds the release bundle under `release_package/`.
- Generates `release_manifest.csv`, `checksums.txt`, `environment_snapshot.txt`, `directory_tree.txt`, `README_RELEASE.md`, and `RUNBOOK.md`.
- Writes the Step-12 JSON summary report to `reports/12_reproducibility_and_release_report.json`.
- Saves the exact config snapshot to `configs_used/12_package_reproducibility_and_release_config.yaml`.
- Optionally creates release archives in `release_archives/`.
- Packages the project outputs only; it does **not** retrain models, reevaluate benchmarks, or regenerate scientific results.

### Main outputs

- `release_package/`
- `release_package/release_manifest.csv`
- `release_package/checksums.txt`
- `release_package/environment_snapshot.txt`
- `release_package/directory_tree.txt`
- `release_package/README_RELEASE.md`
- `release_package/RUNBOOK.md`
- `reports/12_reproducibility_and_release_report.json`
- `logs/12_package_reproducibility_and_release_<timestamp>.log`
- `configs_used/12_package_reproducibility_and_release_config.yaml`
- `release_archives/kinase_causality_qsar_release.tar.gz` when archive creation is enabled

### Example command

```bash
python scripts/12_package_reproducibility_and_release.py --config config.yaml
```

### Reproducibility note

Step 12 is a packaging and archival step only. It preserves path consistency and provenance by copying validated upstream outputs into a release-ready structure and recording manifest/checksum metadata for later verification.
