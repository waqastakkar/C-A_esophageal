# C-A_esophageal

## Project overview

This repository implements a staged TCGA ESCA analysis workflow for tumor-vs-normal modeling.
The study is organized as:

1. **Stage I:** cohort curation and preprocessing.
2. **Stage II:** deep invariant sparse modeling.
3. **Stage III:** candidate driver prioritization and downstream validation.

Step 3 is the bridge between Stage I and Stage II. It establishes a leakage-safe, patient-grouped, manuscript-grade baseline benchmarking framework before the sparse invariant adversarial model is introduced.

## Pipeline overview

1. **Step 1 – generic preprocessing scaffold:** reusable loading, harmonization, scaling, and exploratory train/test export utilities.
2. **Step 2 – curated primary cohort construction:** explicit tumor-vs-normal label creation, aligned primary matrices, and grouped outer-fold creation.
3. **Step 3 – grouped nested baseline benchmarking:** grouped nested cross-validation, strong baseline classifiers, pooled out-of-fold predictions, subgroup analyses, and publication-grade figures.
4. **Step 4 – next planned step:** sparse invariant adversarial model training on the curated Step 2 cohort with the grouped evaluation backbone defined in Step 3.

## Figure Style Standard

All figures in this project must follow the shared publication rules below.

- **SVG is mandatory** for primary figure export.
- **All text must be bold**.
- **Roman/serif fonts only** with preferred order: Times New Roman → Times → STIXGeneral → DejaVu Serif.
- **Reuse `publication_style.py`** for plotting configuration and figure saving.
- **Use the shared Nature-style palette** from `publication_style.py` rather than local ad hoc colors.
- Keep figures manuscript-ready: compact legends, thicker axes, minimal chart junk, balanced whitespace, and editable SVG text where possible.

## Project structure suggestion

```text
C-A_esophageal/
├── preprocess_esca.py
├── step2_build_cohort.py
├── step3_grouped_baselines.py
├── publication_style.py
├── README.md
└── outputs/
    ├── step1_preprocess/
    ├── step2_cohort/
    └── step3_baselines/
```

## Dependency notes

Core dependencies:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib

Optional Step 3 accelerators:

- xgboost
- lightgbm

If XGBoost and LightGBM are unavailable, Step 3 automatically falls back to `HistGradientBoostingClassifier`, so the benchmark remains runnable in a clean environment.

## Step 1 – generic preprocessing scaffold

### Purpose

Step 1 provides a reusable preprocessing scaffold for loading counts, metadata, and normalized matrices; harmonizing identifiers; cleaning missing values; scaling features; and exporting generic modeling tables.

### Key inputs

- `TCGA_ESCA_STAR_Counts.csv`
- `TCGA_ESCA_Metadata.csv`
- `ESCA_vst_normalized_matrix.csv`

### Main logic

- Reads counts, metadata, and normalized matrices.
- Detects and harmonizes sample identifiers.
- Handles gene-by-sample vs sample-by-gene orientation.
- Imputes missing values and applies optional scaling.
- Detects a label column automatically if requested.
- Exports a generic sample-level train/test split.

### Outputs

Representative Step 1 outputs include:

- `TCGA_ESCA_preprocessed.csv`
- `TCGA_ESCA_X_train.csv`
- `TCGA_ESCA_X_test.csv`
- `TCGA_ESCA_y_train.csv`
- `TCGA_ESCA_y_test.csv`
- missingness, outlier, and unmatched-ID diagnostics

### Example command

```bash
python preprocess_esca.py \
  --counts TCGA_ESCA_STAR_Counts.csv \
  --metadata TCGA_ESCA_Metadata.csv \
  --normalized ESCA_vst_normalized_matrix.csv \
  --output-dir outputs/step1_preprocess
```

### Study design note

Step 1 is a **generic scaffold only**. Its auto-detected label column logic and plain `train_test_split` exports are useful for exploratory preprocessing but are **not** the final leakage-safe training source for the TCGA ESCA tumor-vs-normal study.

## Step 2 – curated primary cohort construction

### Purpose

Step 2 converts the generic preprocessing scaffold into a study-grounded primary cohort with explicit tumor-vs-normal labels and leakage-safe grouped outer folds.

### Key inputs

- `TCGA_ESCA_STAR_Counts.csv`
- `ESCA_vst_normalized_matrix.csv`
- `TCGA_ESCA_Metadata.csv`

### Main logic

- Harmonizes TCGA identifiers across all input tables.
- Restricts the cohort to supported primary tumor and solid tissue normal samples.
- Creates explicit binary labels: tumor = 1, normal = 0.
- Builds curated metadata fields for downstream subgroup analyses.
- Selects one representative primary sample per patient for grouped evaluation.
- Generates patient-grouped outer folds for later nested cross-validation.
- Exports aligned primary normalized and counts matrices.

### Outputs

- `master_samples_primary.csv`
- `normalized_primary_matrix.csv`
- `counts_primary_matrix.csv`
- `grouped_outer_folds.csv`
- cohort QC figures in SVG format

### Example command

```bash
python step2_build_cohort.py \
  --counts TCGA_ESCA_STAR_Counts.csv \
  --normalized ESCA_vst_normalized_matrix.csv \
  --metadata TCGA_ESCA_Metadata.csv \
  --output-dir outputs/step2_cohort
```

### Leakage prevention note

Step 2 is the **true modeling entry point** for later stages. Step 3 and future model-training steps must consume the curated Step 2 cohort and grouped folds instead of the generic Step 1 train/test split.

## Step 3 – grouped nested baseline benchmarking

### Purpose

Step 3 benchmarks strong baseline classifiers for TCGA ESCA tumor-vs-normal classification using the curated Step 2 primary cohort and the normalized primary expression matrix. It provides the grouped nested cross-validation backbone that later sparse invariant modeling will inherit.

### Key inputs

- `outputs/step2_cohort/master_samples_primary.csv`
- `outputs/step2_cohort/normalized_primary_matrix.csv`
- `outputs/step2_cohort/grouped_outer_folds.csv`

### Main logic

- Uses **explicit** `disease_label` values from Step 2 rather than auto-detected labels.
- Aligns the normalized matrix to the curated primary cohort by `sample_id`.
- Restricts evaluation to samples shared across the cohort table, normalized matrix, and grouped fold file.
- Enforces **patient-grouped outer folds only** using the precomputed Step 2 fold assignments.
- Builds grouped inner CV within each outer training split for leakage-safe tuning.
- Applies feature filtering, variable-gene selection, imputation, and z-score scaling using **training data only** inside each outer fold.
- Benchmarks these strong baseline models:
  - `LR_L1`
  - `LR_ElasticNet`
  - `RandomForest`
  - `XGBoost` if requested and installed
  - `LightGBM` if requested and available when XGBoost is not used
  - `HistGB` as the mandatory clean-environment fallback
  - `MLP` optionally via `--run-mlp`
- Saves fold metrics, pooled out-of-fold predictions, subgroup summaries, calibration data, and manuscript-grade SVG figures.
- Selects a baseline reference winner using AUROC first, then AUPRC, balanced accuracy, MCC, calibration, and stability.

### Outputs

- `baseline_fold_metrics.csv`
- `baseline_summary_metrics.csv`
- `baseline_oof_predictions.csv`
- `baseline_subgroup_metrics.csv`
- `model_selection_log.csv`
- `baseline_feature_space_summary.csv`
- `calibration_data.csv`
- `run_config.json`
- optional saved fold models when `--save-fold-models` is enabled
- SVG figures including:
  - `figure4_baseline_roc.svg`
  - `figure4_baseline_pr.svg`
  - `figure4_baseline_performance_bar.svg`
  - `figure4_calibration.svg`
  - `figure4_subgroup_performance.svg`
  - `figure4_confusion_summary.svg`
  - `figure4_fold_distribution.svg`

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

### Leakage prevention note

Step 3 explicitly forbids the following:

- no sample-level random train/test split
- no patient overlap between train and validation/test partitions
- no preprocessing fitted on outer test data
- no variable-gene selection using outer test data
- no hyperparameter tuning on outer test data
- no metadata leakage into the default expression-only model inputs

### What Step 3 intentionally does not implement yet

The following are intentionally **out of scope** at Step 3:

- sparse invariant adversarial modeling
- attribution and perturbation analysis
- differential expression
- enrichment analysis
- candidate driver prioritization and validation

## Step 4 – next planned step

Step 4 will implement the **sparse invariant adversarial model** on top of the curated Step 2 cohort and the grouped evaluation backbone created in Step 3.
