# C-A_esophageal

This repository now contains a reusable preprocessing pipeline for the TCGA ESCA gene expression and clinical metadata files used in downstream deep learning experiments.

## Files

- `preprocess_esca.py`: End-to-end preprocessing script for loading, cleaning, normalizing, merging, filtering, and exporting ESCA datasets.

## Expected inputs

Place these CSV files in the repository root, or pass explicit paths on the command line:

- `TCGA_ESCA_STAR_Counts.csv`
- `TCGA_ESCA_Metadata.csv`
- `ESCA_vst_normalized_matrix.csv`

## What the script does

1. Loads all three CSV files into pandas DataFrames.
2. Prints data types and previews the first rows of each table.
3. Detects a shared identifier column automatically, or uses `--id-column` when provided.
4. Imputes missing numeric gene expression values with per-gene means.
5. Cleans metadata by dropping rows with excessive missingness and imputing remaining numeric/categorical fields.
6. Uses the normalized matrix when available; otherwise it falls back to counts and applies `log2(x + 1)` normalization when needed.
7. Optionally keeps only the top percentage of genes ranked by variance.
8. Removes sample outliers using the maximum absolute z-score across gene features.
9. Saves cleaned intermediate tables plus the merged modeling dataset.

## Usage

```bash
python preprocess_esca.py \
  --counts TCGA_ESCA_STAR_Counts.csv \
  --metadata TCGA_ESCA_Metadata.csv \
  --normalized ESCA_vst_normalized_matrix.csv \
  --output-dir processed \
  --gene-top-percent 10
```

### Optional arguments

- `--id-column`: explicitly set the common identifier column.
- `--metadata-missing-threshold`: maximum allowed row-wise metadata missing fraction before a row is dropped. Default: `0.5`.
- `--zscore-threshold`: threshold for sample-level outlier removal. Default: `3.5`.
- `--force-log2-normalization`: force `log2(x + 1)` normalization even when the selected expression matrix already looks normalized.

## Outputs

The script writes the following files into the output directory:

- `TCGA_ESCA_counts_imputed.csv`
- `TCGA_ESCA_metadata_cleaned.csv`
- `TCGA_ESCA_expression_processed.csv`
- `TCGA_ESCA_preprocessed.csv`
- `TCGA_ESCA_missing_summary.csv`
- `TCGA_ESCA_preprocessing_summary.csv`
