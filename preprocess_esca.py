#!/usr/bin/env python3
"""Preprocess TCGA ESCA expression and metadata tables for downstream modeling."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_ID_CANDIDATES = (
    "barcode",
    "sample_submitter_id",
    "bcr_patient_barcode",
    "submitter_id",
    "patient",
    "sample_id",
    "patient_id",
    "patientid",
    "sampleid",
    "case_id",
)

TCGA_IDENTIFIER_PRIORITY = DEFAULT_ID_CANDIDATES
TCGA_BARCODE_COLUMNS = ("barcode", "sample_submitter_id", "submitter_id", "bcr_patient_barcode", "patient")
HARMONIZED_ID_COLUMNS = {"_harmonized_id", "_harmonized_sample_id", "_harmonized_patient_id"}

DEFAULT_LABEL_CANDIDATES = (
    "vital_status",
    "survival_status",
    "tumor_grade",
    "grade",
    "sample_type",
    "project_id",
    "primary_diagnosis",
    "state",
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
        "--label-column",
        default=None,
        help="Metadata column to use as the prediction target. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of merged samples to reserve for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    parser.add_argument(
        "--scale-method",
        choices=("none", "zscore", "minmax"),
        default="zscore",
        help="Feature scaling method applied to gene expression features after merging.",
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
        help="Threshold used when computing the per-sample fraction of genes with extreme absolute z-scores.",
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
    result = df.copy()
    result.columns = [str(col).strip() for col in result.columns]
    return result


def normalize_identifier_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace(".", "-").upper()


def tcga_identifier_variants(value: object) -> dict[str, str | None]:
    normalized = normalize_identifier_value(value)
    variants: dict[str, str | None] = {
        "full": normalized,
        "sample": None,
        "patient": None,
    }
    if not normalized or not normalized.startswith("TCGA-"):
        return variants

    parts = normalized.split("-")
    if len(parts) >= 4:
        variants["sample"] = "-".join(parts[:4])
    if len(parts) >= 3:
        variants["patient"] = "-".join(parts[:3])
    return variants


class DuplicateColumnResolution(NamedTuple):
    dataframe: pd.DataFrame
    duplicate_columns: int
    conflicting_columns: int
    ignored_conflicts: list[str]


def resolve_duplicate_columns(df: pd.DataFrame, preferred_columns: Iterable[str] = ()) -> DuplicateColumnResolution:
    result = df.copy()
    duplicate_columns = 0
    conflicting_columns = 0
    ignored_conflicts: list[str] = []

    for column_name in pd.unique(result.columns[result.columns.duplicated(keep=False)]):
        positions = [idx for idx, col in enumerate(result.columns) if col == column_name]
        if len(positions) < 2:
            continue
        duplicate_columns += len(positions) - 1
        duplicate_frame = result.iloc[:, positions].astype("string")
        normalized = duplicate_frame.applymap(normalize_identifier_value)
        comparable = normalized.fillna(duplicate_frame.apply(lambda col: col.str.strip(), axis=0))
        conflicting_rows = comparable.nunique(axis=1, dropna=True) > 1
        conflicting = bool(conflicting_rows.any())
        if conflicting:
            conflicting_columns += 1
            if column_name in set(preferred_columns):
                ignored_conflicts.append(column_name)
                print(
                    f"Warning: duplicate column '{column_name}' has conflicting values in "
                    f"{int(conflicting_rows.sum())} rows; keeping the first occurrence because it is preferred for TCGA merging."
                )
            else:
                print(
                    f"Warning: duplicate column '{column_name}' has conflicting values in "
                    f"{int(conflicting_rows.sum())} rows; keeping the first occurrence and ignoring the rest."
                )
        else:
            print(
                f"Resolved {len(positions)} duplicated '{column_name}' columns by dropping identical copies."
            )

        kept = duplicate_frame.bfill(axis=1).iloc[:, 0].astype("string")
        result.iloc[:, positions[0]] = kept
        keep_mask = np.ones(len(result.columns), dtype=bool)
        keep_mask[positions[1:]] = False
        result = result.loc[:, keep_mask].copy()

    return DuplicateColumnResolution(result, duplicate_columns, conflicting_columns, ignored_conflicts)


def ensure_unique_identifier_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    resolution = resolve_duplicate_columns(df, preferred_columns=TCGA_IDENTIFIER_PRIORITY)
    result = resolution.dataframe
    matching_positions = [idx for idx, col in enumerate(result.columns) if col == id_column]
    if not matching_positions:
        raise KeyError(f"Identifier column '{id_column}' not found.")

    identifier_values = result.iloc[:, matching_positions].astype("string")
    resolved_identifier = identifier_values.bfill(axis=1).iloc[:, 0].astype("string")
    result.iloc[:, matching_positions[0]] = resolved_identifier
    if len(matching_positions) > 1:
        keep_mask = np.ones(len(result.columns), dtype=bool)
        keep_mask[matching_positions[1:]] = False
        result = result.loc[:, keep_mask].copy()
    return result


def add_harmonized_identifiers(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    result = ensure_unique_identifier_column(df, id_column)
    variants = result[id_column].map(tcga_identifier_variants)
    result["_harmonized_id"] = variants.map(lambda item: item["full"]).astype("string")
    result["_harmonized_sample_id"] = variants.map(lambda item: item["sample"]).astype("string")
    result["_harmonized_patient_id"] = variants.map(lambda item: item["patient"]).astype("string")
    return result


def looks_like_sample_identifier(value: object) -> bool:
    normalized = normalize_identifier_value(value)
    return bool(normalized and normalized.startswith("TCGA-"))


def maybe_transpose_expression_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df.columns) < 3:
        return df

    first_column = df.columns[0]
    other_columns = [col for col in df.columns[1:] if looks_like_sample_identifier(col)]
    if len(other_columns) < max(3, len(df.columns[1:]) // 2):
        return df

    transposed = df.set_index(first_column).transpose().reset_index()
    transposed = transposed.rename(columns={"index": "barcode"})
    transposed.columns = [str(col).strip() for col in transposed.columns]
    print(
        "Detected gene-by-sample expression matrix; transposed to sample-by-gene format "
        f"using '{first_column}' as the feature identifier column."
    )
    return transposed


def canonicalize_sample_identifier_column(df: pd.DataFrame) -> pd.DataFrame:
    resolution = resolve_duplicate_columns(df, preferred_columns=TCGA_IDENTIFIER_PRIORITY)
    result = resolution.dataframe
    has_sample_id = "sample_id" in result.columns
    has_barcode = "barcode" in result.columns

    if has_sample_id and has_barcode:
        print(
            "Warning: both 'sample_id' and 'barcode' are present. "
            "For TCGA expression merging, 'sample_id' is likely UUID-based metadata and 'barcode' will be preferred."
        )

    for preferred_name in TCGA_IDENTIFIER_PRIORITY:
        if preferred_name in result.columns:
            return result

    for column in result.columns:
        if is_unnamed_column(column) and looks_like_identifier_series(result[column]):
            target_name = "barcode" if not has_barcode else column
            if target_name != column:
                result = result.rename(columns={column: target_name})
                print(f"Renamed '{column}' to '{target_name}' based on TCGA-style identifier values.")
            return result
    return result


def is_unnamed_column(column_name: object) -> bool:
    return bool(re.match(r"^unnamed:\s*\d+$", str(column_name).strip(), flags=re.IGNORECASE))


def looks_like_identifier_series(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty:
        return False

    normalized = non_missing.map(normalize_identifier_value).dropna()
    if normalized.empty:
        return False

    uniqueness_ratio = normalized.nunique(dropna=True) / len(normalized)
    if uniqueness_ratio < 0.8:
        return False

    has_identifier_pattern = normalized.astype(str).str.contains(r"[A-Z]", regex=True).any()
    if has_identifier_pattern:
        return True

    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    if numeric_values.notna().all():
        sorted_values = numeric_values.sort_values(ignore_index=True)
        if len(sorted_values) > 1 and sorted_values.diff().dropna().eq(1).all():
            return False

    return True


def detect_identifier(dfs: Iterable[pd.DataFrame], user_choice: str | None = None) -> str:
    dfs = tuple(dfs)
    if user_choice:
        missing = [idx for idx, df in enumerate(dfs, start=1) if user_choice not in df.columns]
        if missing:
            raise KeyError(f"Provided id column '{user_choice}' missing from tables: {missing}")
        return user_choice

    shared = set.intersection(*(set(df.columns) for df in dfs))
    valid_shared = {col for col in shared if not is_unnamed_column(col)}
    for candidate in TCGA_IDENTIFIER_PRIORITY:
        if candidate in valid_shared:
            return candidate
    if len(valid_shared) == 1:
        return next(iter(valid_shared))

    unnamed_shared = sorted(col for col in shared if is_unnamed_column(col))
    if len(unnamed_shared) == 1:
        unnamed_candidate = unnamed_shared[0]
        if all(looks_like_identifier_series(df[unnamed_candidate]) for df in dfs):
            print(
                "Falling back to shared index-like identifier column "
                f"'{unnamed_candidate}' because its values look like sample identifiers."
            )
            return unnamed_candidate

    raise KeyError(
        "Unable to auto-detect a common identifier column after excluding index-like columns. "
        f"Shared columns were: {sorted(shared)}. Use --id-column explicitly."
    )


def log_duplicate_column_summary(dataset_name: str, resolution: DuplicateColumnResolution) -> None:
    print(
        f"{dataset_name}: found {resolution.duplicate_columns} duplicate columns; "
        f"conflicting duplicates ignored: {resolution.conflicting_columns}."
    )
    if resolution.ignored_conflicts:
        print(f"{dataset_name}: conflicting preferred identifier columns ignored: {sorted(set(resolution.ignored_conflicts))}")


def merge_expression_with_metadata(
    expression_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]:
    strategies = [
        ("exact barcode match", "_harmonized_id", "_harmonized_id"),
        ("sample barcode match", "_harmonized_sample_id", "_harmonized_sample_id"),
    ]
    if id_column in {"patient", "bcr_patient_barcode", "patient_id", "patientid"}:
        strategies.append(("patient barcode fallback", "_harmonized_patient_id", "_harmonized_patient_id"))

    for strategy_name, expression_key, metadata_key in strategies:
        left = expression_df[expression_key].dropna()
        right = metadata_df[metadata_key].dropna()
        if left.empty or right.empty:
            continue
        matchable = int(left.isin(set(right)).sum())
        if matchable == 0:
            continue

        unmatched_expression_ids = expression_df.loc[
            ~expression_df[expression_key].isin(set(right)), [id_column, expression_key]
        ].copy()
        unmatched_metadata_ids = metadata_df.loc[
            ~metadata_df[metadata_key].isin(set(left)), [id_column, metadata_key]
        ].copy()

        merged_df = expression_df.merge(
            metadata_df.drop(columns=[id_column]),
            left_on=expression_key,
            right_on=metadata_key,
            how="inner",
            validate="one_to_one",
            suffixes=("", "_metadata"),
        )
        merged_df = merged_df.rename(columns={f"{expression_key}_x": expression_key}) if f"{expression_key}_x" in merged_df.columns else merged_df
        if id_column not in merged_df.columns:
            merged_df[id_column] = merged_df[expression_key]
        return merged_df, unmatched_expression_ids, unmatched_metadata_ids, strategy_name, matchable

    raise ValueError(
        "Unable to merge expression and metadata tables with exact barcode, sample-level harmonization, or patient-level fallback."
    )


def detect_label_column(df: pd.DataFrame, id_column: str, user_choice: str | None) -> str:
    if user_choice:
        if user_choice not in df.columns:
            raise KeyError(f"Provided label column '{user_choice}' not found in metadata.")
        if user_choice == id_column:
            raise ValueError("--label-column cannot match the identifier column.")
        return user_choice

    for candidate in DEFAULT_LABEL_CANDIDATES:
        if candidate in df.columns:
            non_missing = df[candidate].dropna()
            if not non_missing.empty and non_missing.nunique() > 1:
                return candidate

    fallback = [col for col in df.columns if col not in ({id_column} | HARMONIZED_ID_COLUMNS) and df[col].dropna().nunique() > 1]
    if not fallback:
        raise ValueError("Unable to detect a usable label column. Provide --label-column explicitly.")
    return fallback[0]


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


def summarize_missing_by_column(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": df.columns,
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_fraction": [float(df[col].isna().mean()) for col in df.columns],
            "dtype": [str(df[col].dtype) for col in df.columns],
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def clean_metadata(df: pd.DataFrame, id_column: str, missing_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    clinical_cols = [col for col in result.columns if col not in ({id_column} | HARMONIZED_ID_COLUMNS)]
    if not clinical_cols:
        empty_summary = pd.DataFrame(columns=["column", "strategy", "filled_values", "dropped_rows"])
        return result, empty_summary

    missing_fraction = result[clinical_cols].isna().mean(axis=1)
    dropped_mask = missing_fraction > missing_threshold
    dropped_rows = int(dropped_mask.sum())
    result = result.loc[~dropped_mask].copy()

    numeric_cols = [col for col in clinical_cols if pd.api.types.is_numeric_dtype(result[col])]
    categorical_cols = [col for col in clinical_cols if col not in numeric_cols]

    imputation_records: list[dict[str, object]] = []
    for col in numeric_cols:
        missing_before = int(result[col].isna().sum())
        median = result[col].median()
        result[col] = result[col].fillna(median)
        imputation_records.append(
            {"column": col, "strategy": "median", "filled_values": missing_before, "dropped_rows": dropped_rows}
        )
    for col in categorical_cols:
        missing_before = int(result[col].isna().sum())
        mode = result[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        result[col] = result[col].fillna(fill_value)
        imputation_records.append(
            {"column": col, "strategy": f"mode:{fill_value}", "filled_values": missing_before, "dropped_rows": dropped_rows}
        )

    summary = pd.DataFrame(imputation_records).sort_values("filled_values", ascending=False).reset_index(drop=True)
    return result, summary


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


def scale_expression(df: pd.DataFrame, gene_cols: list[str], method: str) -> tuple[pd.DataFrame, str]:
    result = df.copy()
    if method == "none":
        return result, "no scaling"

    values = result[gene_cols].astype(float)
    if method == "zscore":
        std = values.std(axis=0, ddof=0).replace(0, np.nan)
        scaled = ((values - values.mean(axis=0)) / std).fillna(0.0)
        description = "z-score"
    else:
        min_vals = values.min(axis=0)
        ranges = (values.max(axis=0) - min_vals).replace(0, np.nan)
        scaled = ((values - min_vals) / ranges).fillna(0.0)
        description = "min-max"
    result[gene_cols] = scaled
    return result, description


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


def build_outlier_summary(df: pd.DataFrame, gene_cols: list[str], z_threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    if not gene_cols:
        empty = pd.DataFrame(columns=["_harmonized_id", "extreme_fraction", "is_outlier"])
        return empty, pd.Series(dtype=float)

    values = df[gene_cols].astype(float)
    std = values.std(axis=0, ddof=0).replace(0, np.nan)
    zscores = ((values - values.mean(axis=0)) / std).abs().fillna(0.0)
    extreme_fraction = (zscores > z_threshold).mean(axis=1)
    summary = pd.DataFrame(
        {
            "_harmonized_id": df["_harmonized_id"],
            "extreme_fraction": extreme_fraction,
            "is_outlier": extreme_fraction > 0.01,
        }
    )
    return summary, extreme_fraction


def remove_outliers(df: pd.DataFrame, gene_cols: list[str], z_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_before = len(df)
    summary, extreme_fraction = build_outlier_summary(df, gene_cols, z_threshold)
    if summary.empty:
        print(f"Rows before outlier removal: {rows_before}")
        print(f"Rows after outlier removal: {rows_before}")
        print("Samples removed during outlier removal: 0")
        return df.copy(), summary
    filtered = df.loc[extreme_fraction <= 0.01].copy()
    rows_after = len(filtered)
    removed_count = rows_before - rows_after
    print(f"Rows before outlier removal: {rows_before}")
    print(f"Rows after outlier removal: {rows_after}")
    print(f"Samples removed during outlier removal: {removed_count}")
    if rows_before > 0 and rows_after == 0:
        raise ValueError(
            "Outlier filtering removed all merged samples. Adjust the outlier threshold or review the normalized expression matrix."
        )
    return filtered, summary


def encode_metadata_features(df: pd.DataFrame, id_column: str, label_column: str, gene_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    metadata_feature_cols = [
        col for col in df.columns if col not in set(gene_cols) | HARMONIZED_ID_COLUMNS | {id_column, label_column}
    ]
    if not metadata_feature_cols:
        return df.copy(), []

    encoded = pd.get_dummies(df[metadata_feature_cols], dummy_na=False, drop_first=False)
    result = pd.concat([df.drop(columns=metadata_feature_cols), encoded], axis=1)
    return result, encoded.columns.tolist()


def split_train_test(
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError(
            "No samples remain after merging and filtering, so train/test splits cannot be created. "
            "Check the identifier column selection and the unmatched-ID diagnostics."
        )

    feature_cols = [col for col in df.columns if col not in HARMONIZED_ID_COLUMNS | {id_column, label_column}]
    X = df[[id_column, "_harmonized_id"] + feature_cols].copy()
    y = df[[id_column, "_harmonized_id", label_column]].copy()

    if len(df) < 2:
        raise ValueError(
            f"At least 2 merged samples are required for train/test splitting, but only {len(df)} sample remains."
        )

    stratify = None
    label_non_missing = y[label_column].dropna()
    if not label_non_missing.empty and label_non_missing.nunique() > 1 and label_non_missing.value_counts().min() >= 2:
        stratify = y[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return (
        X_train.sort_values(by=id_column).reset_index(drop=True),
        X_test.sort_values(by=id_column).reset_index(drop=True),
        y_train.sort_values(by=id_column).reset_index(drop=True),
        y_test.sort_values(by=id_column).reset_index(drop=True),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_df = maybe_transpose_expression_table(normalize_column_names(load_csv(args.counts)))
    metadata_df = normalize_column_names(load_csv(args.metadata))
    normalized_df = maybe_transpose_expression_table(normalize_column_names(load_csv(args.normalized)))

    counts_resolution = resolve_duplicate_columns(counts_df, preferred_columns=TCGA_IDENTIFIER_PRIORITY)
    metadata_resolution = resolve_duplicate_columns(metadata_df, preferred_columns=TCGA_IDENTIFIER_PRIORITY)
    normalized_resolution = resolve_duplicate_columns(normalized_df, preferred_columns=TCGA_IDENTIFIER_PRIORITY)
    log_duplicate_column_summary("counts", counts_resolution)
    log_duplicate_column_summary("metadata", metadata_resolution)
    log_duplicate_column_summary("normalized", normalized_resolution)

    counts_df = canonicalize_sample_identifier_column(counts_resolution.dataframe)
    metadata_df = canonicalize_sample_identifier_column(metadata_resolution.dataframe)
    normalized_df = canonicalize_sample_identifier_column(normalized_resolution.dataframe)

    id_column = detect_identifier((counts_df, metadata_df, normalized_df), args.id_column)
    print(f"\nSelected merge key: {id_column}")

    counts_df = add_harmonized_identifiers(counts_df.drop_duplicates(subset=id_column), id_column)
    metadata_df = add_harmonized_identifiers(metadata_df.drop_duplicates(subset=id_column), id_column)
    normalized_df = add_harmonized_identifiers(normalized_df.drop_duplicates(subset=id_column), id_column)

    counts_gene_cols, _ = split_expression_columns(counts_df.drop(columns=list(HARMONIZED_ID_COLUMNS)), id_column)
    normalized_gene_cols, _ = split_expression_columns(normalized_df.drop(columns=list(HARMONIZED_ID_COLUMNS)), id_column)

    counts_missing_by_column = summarize_missing_by_column(counts_df, "counts")
    metadata_missing_by_column = summarize_missing_by_column(metadata_df, "metadata")
    normalized_missing_by_column = summarize_missing_by_column(normalized_df, "normalized")

    counts_df = impute_expression(counts_df, counts_gene_cols)
    normalized_df = impute_expression(normalized_df, normalized_gene_cols)
    metadata_df, metadata_imputation_summary = clean_metadata(metadata_df, id_column, args.metadata_missing_threshold)

    selected_expression = normalized_df if normalized_gene_cols else counts_df
    selected_gene_cols = normalized_gene_cols if normalized_gene_cols else counts_gene_cols

    processed_expression, normalization_method = maybe_normalize_expression(
        selected_expression,
        selected_gene_cols,
        args.force_log2_normalization,
    )
    processed_expression, scaling_method = scale_expression(processed_expression, selected_gene_cols, args.scale_method)

    merged_df, unmatched_expression_ids, unmatched_metadata_ids, merge_strategy, merge_candidate_rows = merge_expression_with_metadata(
        processed_expression,
        metadata_df,
        id_column,
    )
    print(f"Merge strategy used: {merge_strategy}")
    print(
        f"Matched samples after merge: {len(merged_df)} | "
        f"Unmatched expression samples: {len(unmatched_expression_ids)} | "
        f"Unmatched metadata samples: {len(unmatched_metadata_ids)}"
    )

    label_column = detect_label_column(metadata_df, id_column, args.label_column)

    merged_df, selected_gene_cols = select_high_variance_genes(merged_df, selected_gene_cols, args.gene_top_percent)
    merged_df, outlier_summary = remove_outliers(merged_df, selected_gene_cols, args.zscore_threshold)
    removed_outliers = int(outlier_summary["is_outlier"].sum()) if not outlier_summary.empty else 0

    merged_df, encoded_metadata_cols = encode_metadata_features(merged_df, id_column, label_column, selected_gene_cols)
    X_train, X_test, y_train, y_test = split_train_test(
        merged_df,
        id_column,
        label_column,
        args.test_size,
        args.random_state,
    )

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
                "label_column",
                "normalization_method",
                "scaling_method",
                "expression_rows",
                "metadata_rows",
                "merge_candidate_rows",
                "merge_strategy",
                "merged_rows",
                "gene_feature_count",
                "encoded_metadata_feature_count",
                "outliers_removed",
                "train_rows",
                "test_rows",
            ],
            "value": [
                id_column,
                label_column,
                normalization_method,
                scaling_method,
                len(processed_expression),
                len(metadata_df),
                merge_candidate_rows,
                merge_strategy,
                len(merged_df),
                len(selected_gene_cols),
                len(encoded_metadata_cols),
                removed_outliers,
                len(X_train),
                len(X_test),
            ],
        }
    )

    train_dataset = X_train.merge(y_train, on=[id_column, "_harmonized_id"], how="inner", validate="one_to_one")
    test_dataset = X_test.merge(y_test, on=[id_column, "_harmonized_id"], how="inner", validate="one_to_one")

    counts_df.to_csv(output_dir / "TCGA_ESCA_counts_imputed.csv", index=False)
    metadata_df.to_csv(output_dir / "TCGA_ESCA_metadata_cleaned.csv", index=False)
    processed_expression.to_csv(output_dir / "TCGA_ESCA_expression_processed.csv", index=False)
    merged_df.to_csv(output_dir / "TCGA_ESCA_preprocessed.csv", index=False)
    X_train.to_csv(output_dir / "TCGA_ESCA_X_train.csv", index=False)
    X_test.to_csv(output_dir / "TCGA_ESCA_X_test.csv", index=False)
    y_train.to_csv(output_dir / "TCGA_ESCA_y_train.csv", index=False)
    y_test.to_csv(output_dir / "TCGA_ESCA_y_test.csv", index=False)
    train_dataset.to_csv(output_dir / "TCGA_ESCA_train_dataset.csv", index=False)
    test_dataset.to_csv(output_dir / "TCGA_ESCA_test_dataset.csv", index=False)
    missing_summary.to_csv(output_dir / "TCGA_ESCA_missing_summary.csv", index=False)
    pd.concat(
        [counts_missing_by_column, metadata_missing_by_column, normalized_missing_by_column],
        ignore_index=True,
    ).to_csv(output_dir / "TCGA_ESCA_missing_by_column.csv", index=False)
    metadata_imputation_summary.to_csv(output_dir / "TCGA_ESCA_metadata_imputation_summary.csv", index=False)
    outlier_summary.sort_values("extreme_fraction", ascending=False).to_csv(
        output_dir / "TCGA_ESCA_outlier_summary.csv", index=False
    )
    unmatched_expression_ids.to_csv(output_dir / "TCGA_ESCA_unmatched_expression_ids.csv", index=False)
    unmatched_metadata_ids.to_csv(output_dir / "TCGA_ESCA_unmatched_metadata_ids.csv", index=False)
    preprocessing_summary.to_csv(output_dir / "TCGA_ESCA_preprocessing_summary.csv", index=False)

    print("\nPreprocessing complete.")
    print(preprocessing_summary)
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
