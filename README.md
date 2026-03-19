# C-A_esophageal

This repository contains a reusable preprocessing pipeline for the TCGA ESCA gene expression and clinical metadata files used in downstream deep learning experiments.

## Files

- `preprocess_esca.py`: End-to-end preprocessing script for loading, inspecting, cleaning, scaling, merging, filtering, and exporting ESCA datasets plus train/test splits.

## Expected inputs

Place these CSV files in the repository root, or pass explicit paths on the command line:

- `TCGA_ESCA_STAR_Counts.csv`
- `TCGA_ESCA_Metadata.csv`
- `ESCA_vst_normalized_matrix.csv`

## What the script does

1. Loads all three CSV files into pandas DataFrames and prints previews for initial inspection.
2. Detects when counts or normalized expression files are stored as gene-by-sample matrices and automatically transposes them into sample-by-gene format.
3. Detects a shared identifier column automatically, or uses `--id-column` when provided.
4. Harmonizes TCGA-style identifiers so metadata and expression tables can merge even when one file uses sample-level barcodes and another uses patient-level barcodes.
5. Reports missingness by dataset and by column.
6. Imputes missing numeric gene expression values with per-gene means.
7. Cleans metadata by dropping rows with excessive missingness and imputing remaining numeric/categorical fields with median/mode strategies.
8. Uses the normalized matrix when available; otherwise it falls back to counts and applies `log2(x + 1)` normalization when needed.
9. Optionally keeps only the top percentage of genes ranked by variance.
10. Scales gene features using z-score normalization or min-max scaling.
11. Detects sample outliers using the maximum absolute z-score across gene features and removes them when they exceed the chosen threshold.
12. Auto-detects a label column (or uses `--label-column`), one-hot encodes remaining metadata features, and exports train/test splits.
13. Saves cleaned intermediate tables, merge diagnostics, outlier summaries, the merged modeling dataset, and train/test outputs.

## Usage

```bash
python preprocess_esca.py \
  --counts TCGA_ESCA_STAR_Counts.csv \
  --metadata TCGA_ESCA_Metadata.csv \
  --normalized ESCA_vst_normalized_matrix.csv \
  --output-dir processed \
  --label-column vital_status \
  --gene-top-percent 10 \
  --scale-method zscore
```

### Optional arguments

- `--id-column`: explicitly set the common identifier column. Auto-detection ignores index-like columns such as `Unnamed: 0` unless they are the only plausible shared identifiers after any needed expression-table transpose.
- `--label-column`: explicitly choose the target label from metadata.
- `--metadata-missing-threshold`: maximum allowed row-wise metadata missing fraction before a row is dropped. Default: `0.5`.
- `--zscore-threshold`: threshold for sample-level outlier removal. Default: `3.5`.
- `--scale-method`: choose `none`, `zscore`, or `minmax` feature scaling. Default: `zscore`.
- `--test-size`: fraction of merged samples reserved for testing. Default: `0.2`.
- `--random-state`: random seed for the train/test split. Default: `42`.
- `--force-log2-normalization`: force `log2(x + 1)` normalization even when the selected expression matrix already looks normalized.

## Outputs

The script writes the following files into the output directory:

- `TCGA_ESCA_counts_imputed.csv`
- `TCGA_ESCA_metadata_cleaned.csv`
- `TCGA_ESCA_expression_processed.csv`
- `TCGA_ESCA_preprocessed.csv`
- `TCGA_ESCA_X_train.csv`
- `TCGA_ESCA_X_test.csv`
- `TCGA_ESCA_y_train.csv`
- `TCGA_ESCA_y_test.csv`
- `TCGA_ESCA_train_dataset.csv`
- `TCGA_ESCA_test_dataset.csv`
- `TCGA_ESCA_missing_summary.csv`
- `TCGA_ESCA_missing_by_column.csv`
- `TCGA_ESCA_metadata_imputation_summary.csv`
- `TCGA_ESCA_outlier_summary.csv`
- `TCGA_ESCA_unmatched_expression_ids.csv`
- `TCGA_ESCA_unmatched_metadata_ids.csv`
- `TCGA_ESCA_preprocessing_summary.csv`

## Notes

- The repository currently does **not** include the TCGA CSV inputs, so you will need to add them locally before running the pipeline.
- If `merged_rows` was previously reported as `0`, inspect `TCGA_ESCA_unmatched_expression_ids.csv` and `TCGA_ESCA_unmatched_metadata_ids.csv`; the harmonized identifier workflow is designed to surface and fix barcode alignment issues.
