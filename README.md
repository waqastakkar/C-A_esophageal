# C-A_esophageal

## Project overview

This repository implements a staged TCGA ESCA tumor-vs-normal modeling workflow aligned to the study design below:

1. **Stage I — Cohort curation and preprocessing**
2. **Stage II — Deep invariant sparse modeling**
3. **Stage III — Candidate driver prioritization and downstream validation**

Step 4 is now the main Stage II implementation. It trains a compact, leakage-safe **Sparse Invariant Adversarial Autoencoder-Classifier** on the curated Step 2 cohort, evaluates it with grouped nested cross-validation consistent with Step 3, and exports interpretable gate weights, latent representations, and manuscript-grade figures.

## Pipeline overview

1. **Step 1 – generic preprocessing scaffold**
2. **Step 2 – curated primary cohort construction**
3. **Step 3 – grouped nested baseline benchmarking**
4. **Step 4 – sparse invariant adversarial deep disease model**
5. **Step 5 – next planned step: stability, attribution, perturbation, and CDPS gene ranking**

## Figure Style Standard

All figures in this project must follow the shared publication rules below.

- **SVG is mandatory** for primary figure export.
- **All text must be bold**.
- **Roman/serif fonts only** with preferred order: Times New Roman → Times → STIXGeneral → DejaVu Serif.
- **Reuse `publication_style.py`** for plotting configuration and figure saving.
- **Use the shared Nature-style palette** from `publication_style.py` rather than local ad hoc colors.
- Keep figures manuscript-ready: compact legends, thicker axes, minimal chart junk, balanced whitespace, and editable SVG text where possible.

## Suggested project structure

```text
C-A_esophageal/
├── preprocess_esca.py
├── step2_build_cohort.py
├── step3_grouped_baselines.py
├── step4_sparse_invariant_model.py
├── deep_model_utils.py
├── publication_style.py
├── README.md
└── outputs/
    ├── step1_preprocess/
    ├── step2_cohort/
    ├── step3_baselines/
    └── step4_deep_model/
```

## Dependency notes

Core dependencies:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- **PyTorch** for Step 4 deep modeling

Optional dependencies:

- `xgboost`
- `lightgbm`
- `umap-learn` for latent-space visualization in Step 4

If `umap-learn` is unavailable, Step 4 falls back to PCA for the latent-space figures.

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
python preprocess_esca.py   --counts TCGA_ESCA_STAR_Counts.csv   --metadata TCGA_ESCA_Metadata.csv   --normalized ESCA_vst_normalized_matrix.csv   --output-dir outputs/step1_preprocess
```

### Leakage/study-design note

Step 1 is a **generic scaffold only**. Its auto-detected labels and sample-level split are not the final training design for the study.

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
python step2_build_cohort.py   --counts TCGA_ESCA_STAR_Counts.csv   --normalized ESCA_vst_normalized_matrix.csv   --metadata TCGA_ESCA_Metadata.csv   --output-dir outputs/step2_cohort
```

### Leakage/study-design note

Step 2 is the **true entry point** for all later model training and grouped evaluation.

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
python step3_grouped_baselines.py   --cohort outputs/step2_cohort/master_samples_primary.csv   --normalized outputs/step2_cohort/normalized_primary_matrix.csv   --folds outputs/step2_cohort/grouped_outer_folds.csv   --output-dir outputs/step3_baselines   --random-seeds 42,52,62,72,82   --inner-folds 3   --run-mlp   --run-xgboost
```

### Leakage/study-design note

Step 3 forbids sample-level random splits, patient overlap across partitions, outer-test-informed preprocessing, and metadata leakage into the default expression-only predictive input.

## Step 4 – sparse invariant adversarial deep disease model

### Purpose

Implements the main Stage II disease model for this study: a **Sparse Invariant Adversarial Autoencoder-Classifier** for TCGA ESCA tumor-vs-normal classification.

### Why this is the main model

Step 4 combines:

- a **sparse feature gate** for interpretable gene-level weighting,
- a compact **encoder** that maps normalized expression into a disease-relevant latent space,
- a **classifier** for the primary tumor-vs-normal endpoint,
- a **decoder** to preserve latent structure via reconstruction,
- **confounder adversary heads** with gradient reversal to reduce latent confounder signal,
- an **environment risk-variance penalty** to encourage stable classification behavior across metadata-defined environments.

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
- Trains a compact PyTorch model with these loss components:
  - classification loss
  - reconstruction loss
  - sparsity loss on the gate
  - adversarial confounder loss
  - invariance penalty across environments
- Exports out-of-fold predictions, fold metrics, training histories, latent embeddings, gate weights, environment summaries, and baseline-vs-deep comparisons.
- Saves manuscript-grade SVG figures through `publication_style.py`.

### Confounder adversary use

The adversary heads predict available confounder/environment labels such as sex, smoking, histology, stage, and country/region from the latent representation. Gradient reversal pushes the encoder toward representations that remain predictive of disease while being less confounder-dominated.

### Invariance penalty use

The invariance component penalizes variance in environment-wise classification risk across usable metadata-defined environments. This encourages more stable disease prediction across available strata while skipping missing or too-small groups gracefully.

### Sparse gate role

The sparse gate is applied directly to normalized gene-expression inputs before encoding. It provides per-gene weights, supports sparsity regularization, and exports interpretable gate-weight tables for later Step 5 stability/attribution/CDPS work.

### Outputs

Expected Step 4 outputs include:

- `deep_fold_metrics.csv`
- `deep_summary_metrics.csv`
- `deep_oof_predictions.csv`
- `deep_training_history.csv`
- `deep_model_selection_log.csv`
- `gate_weights.csv`
- `latent_embeddings.csv` when enabled
- `environment_performance.csv`
- `invariance_summary.csv`
- `baseline_vs_deep_comparison.csv`
- `run_config.json`
- optional fold checkpoints
- manuscript-grade Step 4 SVG figures such as:
  - `figure5_deep_vs_baselines.svg`
  - `figure5_deep_roc.svg`
  - `figure5_deep_pr.svg`
  - `figure5_calibration.svg`
  - `figure5_latent_space_tumor_normal.svg`
  - `figure5_latent_space_histology.svg`
  - `figure5_latent_space_smoking.svg`
  - `figure5_gate_weight_distribution.svg`
  - `figure5_top_gate_genes.svg`
  - `figure5_environment_robustness.svg`
  - optional `figure5_training_dynamics.svg`

### Example command

```bash
python step4_sparse_invariant_model.py   --cohort outputs/step2_cohort/master_samples_primary.csv   --normalized outputs/step2_cohort/normalized_primary_matrix.csv   --folds outputs/step2_cohort/grouped_outer_folds.csv   --baseline-summary outputs/step3_baselines/baseline_summary_metrics.csv   --baseline-oof outputs/step3_baselines/baseline_oof_predictions.csv   --output-dir outputs/step4_deep_model   --random-seeds 42,52,62,72,82   --inner-folds 3   --latent-dim 32   --top-variable-genes 5000   --save-latent-embeddings   --run-umap
```

### Leakage/study-design note

Step 4 preserves the study’s critical grouped evaluation rules:

- no sample-level random splits,
- no patient overlap between outer train and outer test,
- no preprocessing fitted on outer test data,
- no variable-gene selection using outer test data,
- no early stopping on outer test data,
- no confounder/environment encoding used from outer test during training.

### What Step 4 intentionally defers to Step 5 and later

Step 4 does **not** yet implement:

- final CDPS aggregation/ranking,
- stability ranking and perturbation integration,
- differential expression validation,
- pathway enrichment,
- pathway/network analysis,
- claims of biological causality.

Instead, Step 4 provides a confounder-aware disease model and interpretable intermediate outputs that are designed to feed the later candidate-driver prioritization stage.

## Step 5 – planned next step

Step 5 will focus on **stability, attribution, perturbation, and CDPS gene ranking** using the Step 4 model outputs, while still keeping final DE/enrichment/pathway validation as separate downstream tasks.
