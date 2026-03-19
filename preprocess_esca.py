#!/usr/bin/env python3
"""Preprocess TCGA ESCA expression and metadata tables for downstream modeling."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_ID_CANDIDATES = (
    "patient_id",
    "patientid",
    "sample_id",
    "sampleid",
    "case_id",
    "submitter_id",
    "bcr_patient_barcode",
    "barcode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="TCGA_ESCA_STAR_Counts.csv", help="Raw counts CSV path.")
    parser.add_argument("--metadata", default="TCGA_ESCA_Metadata.csv", help="Clinical metadata CSV path.")
    parser.add_argument(
        "--normalized",
        default="ESCA_vst_normalized_matrix.csv",
        help="Normalized/VST matrix CSV path.",
    )
    parser.add_argument("--output-dir", default="processed", help="Directory for generated CSV files.")
    parser.add_argument(
        "--id-column",
        default=None,
        help="Common identifier column shared by the input tables. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--gene-top-percent",
        type=float,
        default=None,
        help="Optional percent of highest-variance genes to keep (e.g. 10 keeps top 10%%).",
    )
    parser.add_argument(
        "--metadata-missing-threshold",
        type=float,
        default=0.5,
        help="Drop metadata rows when more than this fraction of clinical fields is missing.",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=3.5,
        help="Drop samples whose maximum absolute per-gene z-score exceeds this value.",
    )
    parser.add_argument(
        "--force-log2-normalization",
        action="store_true",
        help="Apply log2(x + 1) normalization to the selected expression matrix even if it appears normalized.",
    )
    return parser.parse_args()


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\nLoaded {path}")
    print(df.dtypes)
    print(df.head())
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def detect_identifier(dfs: Iterable[pd.DataFrame], user_choice: str | None = None) -> str:
    if user_choice:
        missing = [idx for idx, df in enumerate(dfs, start=1) if user_choice not in df.columns]
        if missing:
            raise KeyError(f"Provided id column '{user_choice}' missing from tables: {missing}")
        return user_choice

    shared = set.intersection(*(set(df.columns) for df in dfs))
    for candidate in DEFAULT_ID_CANDIDATES:
        if candidate in shared:
            return candidate
    if len(shared) == 1:
        return next(iter(shared))
    raise KeyError(
        "Unable to auto-detect a common identifier column. "
        f"Shared columns were: {sorted(shared)}. Use --id-column explicitly."
    )


def split_expression_columns(df: pd.DataFrame, id_column: str) -> tuple[list[str], list[str]]:
    numeric_cols = [col for col in df.columns if col != id_column and pd.api.types.is_numeric_dtype(df[col])]
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != id_column]
    return numeric_cols, non_numeric_cols


def impute_expression(df: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    result[gene_cols] = result[gene_cols].apply(pd.to_numeric, errors="coerce")
    means = result[gene_cols].mean(axis=0)
    result[gene_cols] = result[gene_cols].fillna(means)
    return result


def clean_metadata(df: pd.DataFrame, id_column: str, missing_threshold: float) -> pd.DataFrame:
    result = df.copy()
    clinical_cols = [col for col in result.columns if col != id_column]
    if not clinical_cols:
        return result

    missing_fraction = result[clinical_cols].isna().mean(axis=1)
    result = result.loc[missing_fraction <= missing_threshold].copy()

    numeric_cols = [col for col in clinical_cols if pd.api.types.is_numeric_dtype(result[col])]
    categorical_cols = [col for col in clinical_cols if col not in numeric_cols]

    for col in numeric_cols:
        result[col] = result[col].fillna(result[col].median())
    for col in categorical_cols:
        mode = result[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        result[col] = result[col].fillna(fill_value)
    return result


def maybe_normalize_expression(df: pd.DataFrame, gene_cols: list[str], force_log2: bool) -> tuple[pd.DataFrame, str]:
    result = df.copy()
    values = result[gene_cols]
    max_value = float(values.max().max())
    min_value = float(values.min().min())
    appears_normalized = min_value >= 0 and max_value <= 50

    if force_log2 or not appears_normalized:
        result[gene_cols] = np.log2(values + 1.0)
        method = "log2(x + 1)"
    else:
        method = "pre-normalized input retained"
    return result, method


def select_high_variance_genes(df: pd.DataFrame, gene_cols: list[str], top_percent: float | None) -> tuple[pd.DataFrame, list[str]]:
    if top_percent is None:
        return df.copy(), gene_cols
    if not 0 < top_percent <= 100:
        raise ValueError("--gene-top-percent must be within (0, 100].")

    keep_n = max(1, int(np.ceil(len(gene_cols) * (top_percent / 100.0))))
    variances = df[gene_cols].var(axis=0).sort_values(ascending=False)
    selected = variances.head(keep_n).index.tolist()
    other_cols = [col for col in df.columns if col not in gene_cols]
    return df[other_cols + selected].copy(), selected


def remove_outliers(df: pd.DataFrame, gene_cols: list[str], z_threshold: float) -> tuple[pd.DataFrame, int]:
    if not gene_cols:
        return df.copy(), 0

    values = df[gene_cols]
    std = values.std(axis=0, ddof=0).replace(0, np.nan)
    zscores = ((values - values.mean(axis=0)) / std).abs().fillna(0.0)
    mask = zscores.max(axis=1) <= z_threshold
    filtered = df.loc[mask].copy()
    removed = int((~mask).sum())
    return filtered, removed


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_df = normalize_column_names(load_csv(args.counts))
    metadata_df = normalize_column_names(load_csv(args.metadata))
    normalized_df = normalize_column_names(load_csv(args.normalized))

    id_column = detect_identifier((counts_df, metadata_df, normalized_df), args.id_column)
    print(f"\nUsing common identifier column: {id_column}")

    counts_df = counts_df.drop_duplicates(subset=id_column)
    metadata_df = metadata_df.drop_duplicates(subset=id_column)
    normalized_df = normalized_df.drop_duplicates(subset=id_column)

    counts_gene_cols, _ = split_expression_columns(counts_df, id_column)
    normalized_gene_cols, _ = split_expression_columns(normalized_df, id_column)

    counts_df = impute_expression(counts_df, counts_gene_cols)
    normalized_df = impute_expression(normalized_df, normalized_gene_cols)
    metadata_df = clean_metadata(metadata_df, id_column, args.metadata_missing_threshold)

    selected_expression = normalized_df if normalized_gene_cols else counts_df
    selected_gene_cols = normalized_gene_cols if normalized_gene_cols else counts_gene_cols

    processed_expression, normalization_method = maybe_normalize_expression(
        selected_expression,
        selected_gene_cols,
        args.force_log2_normalization,
    )

    merged_df = processed_expression.merge(metadata_df, on=id_column, how="inner", validate="one_to_one")
    merged_df, selected_gene_cols = select_high_variance_genes(merged_df, selected_gene_cols, args.gene_top_percent)
    merged_df, removed_outliers = remove_outliers(merged_df, selected_gene_cols, args.zscore_threshold)

    missing_summary = pd.DataFrame(
        {
            "dataset": ["counts", "metadata", "normalized", "merged"],
            "missing_values": [
                int(counts_df.isna().sum().sum()),
                int(metadata_df.isna().sum().sum()),
                int(normalized_df.isna().sum().sum()),
                int(merged_df.isna().sum().sum()),
            ],
        }
    )

    preprocessing_summary = pd.DataFrame(
        {
            "metric": [
                "identifier_column",
                "normalization_method",
                "expression_rows",
                "metadata_rows",
                "merged_rows",
                "gene_feature_count",
                "outliers_removed",
            ],
            "value": [
                id_column,
                normalization_method,
                len(processed_expression),
                len(metadata_df),
                len(merged_df),
                len(selected_gene_cols),
                removed_outliers,
            ],
        }
    )

    counts_df.to_csv(output_dir / "TCGA_ESCA_counts_imputed.csv", index=False)
    metadata_df.to_csv(output_dir / "TCGA_ESCA_metadata_cleaned.csv", index=False)
    processed_expression.to_csv(output_dir / "TCGA_ESCA_expression_processed.csv", index=False)
    merged_df.to_csv(output_dir / "TCGA_ESCA_preprocessed.csv", index=False)
    missing_summary.to_csv(output_dir / "TCGA_ESCA_missing_summary.csv", index=False)
    preprocessing_summary.to_csv(output_dir / "TCGA_ESCA_preprocessing_summary.csv", index=False)

    print("\nPreprocessing complete.")
    print(preprocessing_summary)
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
