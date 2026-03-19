#!/usr/bin/env python3
"""Build a study-ready TCGA ESCA cohort and grouped fold metadata.

Example usage
-------------
python step2_build_cohort.py \
    --counts TCGA_ESCA_STAR_Counts.csv \
    --normalized ESCA_vst_normalized_matrix.csv \
    --metadata TCGA_ESCA_Metadata.csv \
    --output-dir step2_outputs \
    --outer-folds 5 \
    --random-state 42 \
    --keep-all-samples

Step 3 will plug into the exported cohort tables and patient-grouped fold metadata
for grouped nested cross-validation and baseline model training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from publication_style import (
    apply_text_style,
    configure_publication_plotting,
    get_nature_palette,
    save_publication_figure,
)

DEFAULT_ID_CANDIDATES: tuple[str, ...] = (
    "barcode",
    "sample_submitter_id",
    "submitter_id",
    "case_submitter_id",
    "bcr_patient_barcode",
    "patient",
    "sample_id",
    "patient_id",
    "patientid",
    "sampleid",
    "case_id",
)

CANONICAL_METADATA_COLUMNS: dict[str, tuple[str, ...]] = {
    "sex_raw": ("sex", "gender", "gender.demographic"),
    "race_raw": ("race", "ethnicity", "race_category"),
    "country_raw": (
        "country",
        "country_of_origin",
        "country_of_residence",
        "region",
        "project_country",
        "tumor_tissue_site",
    ),
    "smoking_raw": (
        "smoking_status",
        "tobacco_smoking_history",
        "tobacco_smoking_status",
        "pack_years_smoked",
        "cigarettes_per_day",
    ),
    "pack_years_raw": ("pack_years_smoked", "pack_years", "tobacco_smoking_pack_years_smoked"),
    "alcohol_history_raw": (
        "alcohol_history",
        "alcohol_consumption",
        "alcohol_use",
        "history_of_alcohol_consumption",
    ),
    "stage_raw": (
        "ajcc_pathologic_stage",
        "ajcc_clinical_stage",
        "stage_event_pathologic_stage",
        "tumor_stage",
        "stage",
    ),
    "histology_raw": (
        "primary_diagnosis",
        "histological_type",
        "histologic_diagnosis.1",
        "histology",
        "paper_histological_type",
    ),
    "sample_type_metadata": ("sample_type", "sample_type_name", "definition", "tumor_normal"),
    "survival_status_raw": ("vital_status", "survival_status", "os_event", "death_event"),
    "days_to_death": ("days_to_death", "daystodeath"),
    "days_to_last_follow_up": (
        "days_to_last_follow_up",
        "daystolastfollowup",
        "days_to_last_known_alive",
    ),
    "age_at_diagnosis": (
        "age_at_diagnosis",
        "age_at_initial_pathologic_diagnosis",
        "age",
        "diagnosis_age",
    ),
}

SUPPORTED_PRIMARY_CODES = {"01": "primary_tumor", "11": "solid_tissue_normal"}


@dataclass(frozen=True)
class IdentifierVariants:
    full_barcode: str | None
    sample_barcode: str | None
    patient_barcode: str | None
    sample_type_code: str | None


def normalize_identifier_value(value: object) -> str | None:
    """Normalize identifier-like values into uppercase TCGA-friendly strings."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return None
    return text.replace(".", "-").upper()



def tcga_identifier_variants(value: object) -> dict[str, str | None]:
    """Return full/sample/patient TCGA barcode variants and sample type code."""
    normalized = normalize_identifier_value(value)
    variants = {
        "full_barcode": normalized,
        "sample_barcode": None,
        "patient_barcode": None,
        "sample_type_code": None,
    }
    if not normalized or not normalized.startswith("TCGA-"):
        return variants

    parts = normalized.split("-")
    if len(parts) >= 3:
        variants["patient_barcode"] = "-".join(parts[:3])
    if len(parts) >= 4:
        variants["sample_barcode"] = "-".join(parts[:4])
        sample_field = parts[3]
        match = re.match(r"^(\d{2})", sample_field)
        if match:
            variants["sample_type_code"] = match.group(1)
    return variants



def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from column labels."""
    result = df.copy()
    result.columns = [str(col).strip() for col in result.columns]
    return result



def is_unnamed_column(column_name: object) -> bool:
    """Return True when a column looks like a CSV index artifact."""
    return bool(re.match(r"^unnamed:\s*\d+$", str(column_name).strip(), flags=re.IGNORECASE))



def looks_like_identifier_series(series: pd.Series) -> bool:
    """Heuristic for detecting shared identifier columns."""
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    normalized = non_missing.map(normalize_identifier_value).dropna()
    if normalized.empty:
        return False
    uniqueness_ratio = normalized.nunique(dropna=True) / len(normalized)
    if uniqueness_ratio < 0.8:
        return False
    return normalized.astype(str).str.contains(r"TCGA-|^[A-Z0-9_-]+$", regex=True).any()



def maybe_transpose_expression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose gene-by-sample matrices into sample-by-gene matrices when needed."""
    if df.empty or len(df.columns) < 3:
        return df
    tcga_like_columns = [col for col in df.columns[1:] if normalize_identifier_value(col or "")]
    if len(tcga_like_columns) < max(3, len(df.columns[1:]) // 2):
        return df
    transposed = df.set_index(df.columns[0]).transpose().reset_index()
    transposed = transposed.rename(columns={"index": "barcode"})
    transposed.columns = [str(col).strip() for col in transposed.columns]
    return transposed



def detect_identifier(dfs: Iterable[pd.DataFrame], user_choice: str | None = None) -> str:
    """Detect the shared identifier column across the input tables."""
    dataframes = tuple(dfs)
    if user_choice is not None:
        missing = [index for index, df in enumerate(dataframes, start=1) if user_choice not in df.columns]
        if missing:
            raise KeyError(f"Identifier column '{user_choice}' missing from tables: {missing}")
        return user_choice

    shared_columns = set.intersection(*(set(df.columns) for df in dataframes))
    filtered_shared = {col for col in shared_columns if not is_unnamed_column(col)}
    for candidate in DEFAULT_ID_CANDIDATES:
        if candidate in filtered_shared:
            return candidate
    if len(filtered_shared) == 1:
        return next(iter(filtered_shared))
    unnamed_shared = [col for col in shared_columns if is_unnamed_column(col)]
    if len(unnamed_shared) == 1 and all(looks_like_identifier_series(df[unnamed_shared[0]]) for df in dataframes):
        return unnamed_shared[0]
    raise KeyError(
        "Unable to detect a common identifier column. "
        f"Shared columns were: {sorted(shared_columns)}. Provide --id-column explicitly."
    )



def add_harmonized_identifiers(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Add harmonized TCGA identifier variants to a dataframe."""
    result = df.copy()
    if id_column not in result.columns:
        raise KeyError(f"Identifier column '{id_column}' not found.")
    result[id_column] = result[id_column].map(normalize_identifier_value).astype("string")
    variants = result[id_column].map(tcga_identifier_variants)
    result["_harmonized_id"] = variants.map(lambda item: item["full_barcode"]).astype("string")
    result["_harmonized_sample_id"] = variants.map(lambda item: item["sample_barcode"]).astype("string")
    result["_harmonized_patient_id"] = variants.map(lambda item: item["patient_barcode"]).astype("string")
    result["_sample_type_code"] = variants.map(lambda item: item["sample_type_code"]).astype("string")
    return result



def standardize_metadata_columns(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata column names and expose stable canonical aliases."""
    result = metadata_df.copy()
    normalized_map = {
        column: re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") for column in result.columns
    }
    result = result.rename(columns=normalized_map)
    for canonical_name, synonyms in CANONICAL_METADATA_COLUMNS.items():
        for synonym in synonyms:
            if synonym in result.columns and canonical_name not in result.columns:
                result[canonical_name] = result[synonym]
                break
    return result



def infer_sample_type(metadata_df: pd.DataFrame, tumor_code: str, normal_code: str) -> pd.DataFrame:
    """Infer barcode-derived and metadata-derived sample types, preserving conflicts."""
    result = metadata_df.copy()

    def parse_metadata_sample_type(value: object) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if tumor_code == "01" and ("primary tumor" in text or text in {"tumor", "primary_tumor"}):
            return tumor_code
        if normal_code == "11" and ("solid tissue normal" in text or text in {"normal", "solid_normal", "solid_tissue_normal"}):
            return normal_code
        match = re.match(r"^(\d{2})", text)
        if match:
            return match.group(1)
        return None

    metadata_source = result.get("sample_type_metadata")
    result["sample_type_code_barcode"] = result["_sample_type_code"].astype("string")
    result["sample_type_code_metadata"] = (
        metadata_source.map(parse_metadata_sample_type).astype("string") if metadata_source is not None else pd.Series(pd.NA, index=result.index, dtype="string")
    )
    result["sample_type_conflict"] = (
        result["sample_type_code_barcode"].notna()
        & result["sample_type_code_metadata"].notna()
        & (result["sample_type_code_barcode"] != result["sample_type_code_metadata"])
    )
    result["sample_type_code"] = result["sample_type_code_barcode"].fillna(result["sample_type_code_metadata"]).astype("string")
    result["sample_type_label"] = result["sample_type_code"].map(SUPPORTED_PRIMARY_CODES).fillna("unsupported")
    return result



def build_primary_label(df: pd.DataFrame, tumor_code: str, normal_code: str) -> pd.DataFrame:
    """Build explicit binary tumor/normal labels restricted to supported TCGA codes."""
    result = df.copy()
    label_map = {tumor_code: 1, normal_code: 0}
    name_map = {tumor_code: "tumor", normal_code: "normal"}
    result["disease_label"] = result["sample_type_code"].map(label_map)
    result["disease_label_name"] = result["sample_type_code"].map(name_map)
    result["primary_supported_sample"] = result["sample_type_code"].isin({tumor_code, normal_code})
    return result



def _find_best_available_column(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None



def _clean_unknown(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"not reported", "not available", "unknown", "nan", "none", "null", "--", "[not available]", "[unknown]"}:
        return "unknown"
    return text



def _categorize_smoking(value: object) -> str:
    text = _clean_unknown(value).lower()
    if text == "unknown":
        return "unknown"
    if any(token in text for token in ("never", "non-smoker", "nonsmoker", "lifelong non-smoker", "no")):
        return "non_smoker"
    if any(token in text for token in ("current", "reformed", "former", "smoker", "yes", "1", "2", "3", "4", "5")):
        return "smoker"
    return "unknown"



def _categorize_sex(value: object) -> str:
    text = _clean_unknown(value).lower()
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return "unknown"



def _categorize_histology(value: object) -> str:
    text = _clean_unknown(value).lower()
    if text == "unknown":
        return "unknown"
    if "squamous" in text or "escc" in text:
        return "escc"
    if "adenocarcinoma" in text or "eac" in text or "adeno" in text:
        return "eac"
    return "other"



def _categorize_stage(value: object) -> str:
    text = _clean_unknown(value).lower().replace("stage", "").strip()
    if text == "unknown":
        return "unknown"
    if text.startswith(("i", "1", "ia", "ib", "ii", "2", "iia", "iib")):
        return "early_stage"
    if text.startswith(("iii", "3", "iv", "4")):
        return "late_stage"
    return "unknown"



def _categorize_region(value: object) -> str:
    text = _clean_unknown(value)
    if text == "unknown":
        return text
    compact = re.sub(r"\s+", "_", text.lower())
    return compact[:40]



def build_confounder_table(df: pd.DataFrame, max_missing_fraction: float) -> pd.DataFrame:
    """Build a cleaned confounder table with raw and compact environment variables."""
    result = df.copy()
    required_columns = [
        "_harmonized_patient_id",
        "_harmonized_sample_id",
        "disease_label",
        "disease_label_name",
        "sample_type_code",
        "sample_type_label",
        "sample_type_conflict",
    ]
    for column in required_columns:
        if column not in result.columns:
            raise KeyError(f"Required cohort column '{column}' missing before confounder construction.")

    result["patient_id"] = result["_harmonized_patient_id"].astype("string")
    result["sample_id"] = result["_harmonized_sample_id"].fillna(result["_harmonized_id"]).astype("string")
    result["original_id"] = result["_harmonized_id"].astype("string")

    for raw_column in (
        "histology_raw",
        "sex_raw",
        "smoking_raw",
        "stage_raw",
        "country_raw",
        "race_raw",
        "age_at_diagnosis",
        "alcohol_history_raw",
        "survival_status_raw",
        "days_to_death",
        "days_to_last_follow_up",
        "pack_years_raw",
    ):
        if raw_column not in result.columns:
            result[raw_column] = pd.NA

    result["histology_clean"] = result["histology_raw"].map(_categorize_histology)
    result["env_sex"] = result["sex_raw"].map(_categorize_sex)
    result["env_smoking"] = result["smoking_raw"].map(_categorize_smoking)
    result["env_histology"] = result["histology_raw"].map(_categorize_histology)
    result["env_stage"] = result["stage_raw"].map(_categorize_stage)
    result["env_country_or_region"] = result["country_raw"].map(_categorize_region)

    numeric_candidates = ["age_at_diagnosis", "days_to_death", "days_to_last_follow_up", "pack_years_raw"]
    for column in numeric_candidates:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    metadata_fields = [
        "histology_raw",
        "sex_raw",
        "smoking_raw",
        "stage_raw",
        "country_raw",
        "race_raw",
        "age_at_diagnosis",
        "alcohol_history_raw",
        "survival_status_raw",
        "days_to_death",
        "days_to_last_follow_up",
        "pack_years_raw",
    ]
    result["metadata_completeness_score"] = result[metadata_fields].notna().mean(axis=1)
    result["confounder_missing_fraction"] = 1.0 - result["metadata_completeness_score"]
    result["confounder_missing_excess"] = result["confounder_missing_fraction"] > float(max_missing_fraction)
    return result



def select_representative_samples(
    primary_df: pd.DataFrame,
    prefer_paired_normal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Select one deterministic tumor and one deterministic normal sample max per patient."""
    if primary_df.empty:
        return primary_df.copy(), pd.DataFrame(columns=list(primary_df.columns) + ["exclusion_reason"]), {
            "duplicate_patient_count": 0,
            "duplicate_sample_reductions": 0,
        }

    ranking_df = primary_df.copy()
    ranking_df["_has_full_barcode"] = ranking_df["original_id"].notna().astype(int)
    ranking_df["_sort_sample_id"] = ranking_df["sample_id"].fillna("")
    ranking_df["_priority_paired_normal"] = np.where(
        prefer_paired_normal & ranking_df["disease_label_name"].eq("normal"),
        1,
        0,
    )

    kept_chunks: list[pd.DataFrame] = []
    excluded_rows: list[pd.DataFrame] = []
    duplicate_patient_count = 0
    duplicate_sample_reductions = 0

    for patient_id, patient_frame in ranking_df.groupby("patient_id", sort=True, dropna=False):
        if pd.isna(patient_id):
            raise ValueError("Patient ID derivation failed for at least one sample; cannot continue.")
        if len(patient_frame) > 1:
            duplicate_patient_count += 1

        for label_name in ("tumor", "normal"):
            subset = patient_frame[patient_frame["disease_label_name"] == label_name].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(
                by=[
                    "metadata_completeness_score",
                    "_has_full_barcode",
                    "_priority_paired_normal",
                    "_sort_sample_id",
                ],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            selected = subset.iloc[[0]].copy()
            selected["representative_selected"] = True
            kept_chunks.append(selected)
            dropped = subset.iloc[1:].copy()
            if not dropped.empty:
                duplicate_sample_reductions += len(dropped)
                dropped["representative_selected"] = False
                dropped["exclusion_reason"] = f"duplicate_{label_name}_sample_within_patient"
                excluded_rows.append(dropped)

        unsupported = patient_frame[~patient_frame["disease_label_name"].isin(["tumor", "normal"])].copy()
        if not unsupported.empty:
            unsupported["representative_selected"] = False
            unsupported["exclusion_reason"] = "unsupported_sample_type_for_primary_cohort"
            excluded_rows.append(unsupported)

    kept_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else ranking_df.iloc[0:0].copy()
    kept_df = kept_df.sort_values(["patient_id", "disease_label", "sample_id"], kind="mergesort").reset_index(drop=True)
    excluded_df = pd.concat(excluded_rows, ignore_index=True) if excluded_rows else ranking_df.iloc[0:0].copy()
    return kept_df, excluded_df, {
        "duplicate_patient_count": int(duplicate_patient_count),
        "duplicate_sample_reductions": int(duplicate_sample_reductions),
    }



def _iter_group_stratified_folds(
    y: Sequence[int],
    groups: Sequence[str],
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create grouped stratified folds with sklearn or a deterministic fallback."""
    unique_groups = pd.Index(pd.Series(groups, dtype="string").dropna().unique())
    if len(unique_groups) < n_splits:
        raise ValueError(f"Cannot create {n_splits} outer folds with only {len(unique_groups)} unique patients.")

    group_to_label = {}
    for label_value, group_value in zip(y, groups):
        if group_value in group_to_label and group_to_label[group_value] != label_value:
            raise ValueError(
                "A patient appears with both tumor and normal labels in the primary cohort. "
                "Grouped stratification requires one label per group; paired design should be split in a later step."
            )
        group_to_label[group_value] = int(label_value)

    group_items = sorted(group_to_label.items())
    group_names = np.array([item[0] for item in group_items], dtype=object)
    group_labels = np.array([item[1] for item in group_items], dtype=int)

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        group_frame = pd.DataFrame({"group": group_names, "label": group_labels})
        splits = list(splitter.split(group_frame[["group"]], group_labels, groups=group_names))
        mapped_splits: list[tuple[np.ndarray, np.ndarray]] = []
        sample_groups = np.array(groups, dtype=object)
        for _, test_group_index in splits:
            test_groups = set(group_names[test_group_index])
            test_index = np.flatnonzero(np.isin(sample_groups, list(test_groups)))
            train_index = np.flatnonzero(~np.isin(sample_groups, list(test_groups)))
            mapped_splits.append((train_index, test_index))
        return mapped_splits
    except Exception:
        rng = np.random.default_rng(random_state)
        label_to_groups: dict[int, list[str]] = defaultdict(list)
        for group_name, label in group_items:
            label_to_groups[label].append(group_name)
        for label in label_to_groups:
            label_to_groups[label] = list(np.array(sorted(label_to_groups[label]))[rng.permutation(len(label_to_groups[label]))])
        fold_groups: list[set[str]] = [set() for _ in range(n_splits)]
        for label, label_groups in sorted(label_to_groups.items()):
            for index, group_name in enumerate(label_groups):
                fold_groups[index % n_splits].add(group_name)
        sample_groups = np.array(groups, dtype=object)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_group_set in fold_groups:
            test_index = np.flatnonzero(np.isin(sample_groups, list(fold_group_set)))
            train_index = np.flatnonzero(~np.isin(sample_groups, list(fold_group_set)))
            folds.append((train_index, test_index))
        return folds



def create_grouped_folds(primary_df: pd.DataFrame, outer_folds: int, random_state: int) -> pd.DataFrame:
    """Create leakage-safe grouped outer CV folds at the patient level."""
    if primary_df["patient_id"].duplicated().any():
        duplicated_pairs = primary_df.groupby("patient_id")["disease_label"].nunique()
        if (duplicated_pairs > 1).any():
            raise ValueError(
                "Grouped folds cannot be assigned because at least one patient has both tumor and normal samples in the primary cohort. "
                "Re-run without paired retention for primary cross-validation outputs."
            )
    splits = _iter_group_stratified_folds(
        y=primary_df["disease_label"].astype(int).tolist(),
        groups=primary_df["patient_id"].astype(str).tolist(),
        n_splits=outer_folds,
        random_state=random_state,
    )
    fold_assignments = pd.DataFrame(
        {
            "sample_id": primary_df["sample_id"].astype(str),
            "patient_id": primary_df["patient_id"].astype(str),
            "disease_label": primary_df["disease_label"].astype(int),
            "outer_fold": -1,
        }
    )
    for fold_number, (_, test_index) in enumerate(splits, start=1):
        fold_assignments.loc[test_index, "outer_fold"] = fold_number
    if (fold_assignments["outer_fold"] < 1).any():
        raise RuntimeError("Failed to assign grouped outer folds to every primary sample.")
    return fold_assignments.sort_values(["outer_fold", "sample_id"], kind="mergesort").reset_index(drop=True)



def align_expression_to_cohort(
    expression_df: pd.DataFrame,
    selected_sample_ids: Sequence[str],
    id_column: str,
) -> pd.DataFrame:
    """Restrict an expression matrix to curated cohort samples without mixing matrices."""
    result = add_harmonized_identifiers(expression_df.copy(), id_column)
    result["sample_id"] = result["_harmonized_sample_id"].fillna(result["_harmonized_id"]).astype("string")
    selected_index = pd.Index(pd.Series(selected_sample_ids, dtype="string"))
    missing = selected_index.difference(pd.Index(result["sample_id"].dropna().astype(str)))
    if not missing.empty:
        raise ValueError(f"Expression matrix is missing curated samples: {missing.tolist()[:10]}")
    aligned = result[result["sample_id"].isin(selected_index)].copy()
    aligned = aligned.drop_duplicates(subset="sample_id", keep="first")
    aligned = aligned.set_index("sample_id").reindex(selected_index)
    aligned.index.name = "sample_id"
    aligned = aligned.reset_index()
    if aligned["sample_id"].isna().any():
        raise ValueError("Expression alignment created missing sample rows unexpectedly.")
    drop_columns = [id_column, "_harmonized_id", "_harmonized_sample_id", "_harmonized_patient_id", "_sample_type_code"]
    keep_columns = ["sample_id"] + [column for column in aligned.columns if column not in drop_columns + ["sample_id"]]
    return aligned[keep_columns]



def _summarize_available_confounders(df: pd.DataFrame) -> dict[str, int]:
    fields = [
        "histology_raw",
        "sex_raw",
        "smoking_raw",
        "stage_raw",
        "country_raw",
        "race_raw",
        "age_at_diagnosis",
        "alcohol_history_raw",
        "survival_status_raw",
        "days_to_death",
        "days_to_last_follow_up",
        "pack_years_raw",
    ]
    return {field: int(df[field].notna().sum()) for field in fields if field in df.columns}



def _environment_category_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    environment_columns = ["env_sex", "env_smoking", "env_histology", "env_stage", "env_country_or_region"]
    summary: dict[str, dict[str, int]] = {}
    for column in environment_columns:
        if column in df.columns:
            summary[column] = {str(key): int(value) for key, value in df[column].value_counts(dropna=False).items()}
    return summary



def _create_qc_figures(output_dir: Path, master_all_df: pd.DataFrame, primary_df: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    """Generate cohort/QC SVG figures using the shared publication style."""
    import matplotlib.pyplot as plt

    configure_publication_plotting()
    palette = get_nature_palette()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    label_counts = primary_df["disease_label_name"].value_counts().reindex(["tumor", "normal"]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    ax.bar(label_counts.index, label_counts.values, color=[palette["tumor"], palette["normal"]], width=0.6)
    ax.set_title("Primary cohort class balance")
    ax.set_ylabel("Samples")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "cohort_class_balance.svg")
    plt.close(fig)

    fold_counts = fold_df.groupby(["outer_fold", "disease_label"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    bottom = np.zeros(len(fold_counts), dtype=int)
    for label, color_key, label_name in [(1, "tumor", "Tumor"), (0, "normal", "Normal")]:
        values = fold_counts.get(label, pd.Series(0, index=fold_counts.index)).to_numpy()
        ax.bar(fold_counts.index.astype(str), values, bottom=bottom, color=palette[color_key], label=label_name, width=0.7)
        bottom = bottom + values
    ax.set_title("Grouped outer-fold distribution")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Samples")
    ax.legend(title="Class")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "grouped_fold_distribution.svg")
    plt.close(fig)

    completeness = master_all_df[["metadata_completeness_score"]].copy()
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    ax.hist(completeness["metadata_completeness_score"], bins=10, color=palette["unknown"], edgecolor="white")
    ax.set_title("Metadata completeness")
    ax.set_xlabel("Completeness score")
    ax.set_ylabel("Samples")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "metadata_completeness.svg")
    plt.close(fig)



def write_outputs(
    output_dir: Path,
    master_all_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    normalized_primary: pd.DataFrame,
    counts_primary: pd.DataFrame,
    fold_df: pd.DataFrame,
    exclusion_log_df: pd.DataFrame,
    summary_payload: dict[str, object],
) -> None:
    """Write all cohort deliverables to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    master_all_df.to_csv(output_dir / "master_samples_all.csv", index=False)
    primary_df.to_csv(output_dir / "master_samples_primary.csv", index=False)
    normalized_primary.to_csv(output_dir / "normalized_primary_matrix.csv", index=False)
    counts_primary.to_csv(output_dir / "counts_primary_matrix.csv", index=False)
    fold_df.to_csv(output_dir / "grouped_outer_folds.csv", index=False)
    exclusion_log_df.to_csv(output_dir / "exclusion_log.csv", index=False)
    with (output_dir / "cohort_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
    _create_qc_figures(output_dir, master_all_df, primary_df, fold_df)



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, help="Path to raw counts CSV.")
    parser.add_argument("--normalized", required=True, help="Path to normalized/VST matrix CSV.")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--id-column", default=None, help="Optional explicit identifier column shared by all inputs.")
    parser.add_argument("--tumor-code", default="01", help="TCGA tumor sample type code.")
    parser.add_argument("--normal-code", default="11", help="TCGA normal sample type code.")
    parser.add_argument("--outer-folds", type=int, default=5, help="Number of patient-grouped outer folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic grouped fold creation.")
    parser.add_argument(
        "--max-missing-confounder-fraction",
        type=float,
        default=0.6,
        help="Flag samples with confounder missingness above this fraction.",
    )
    parser.add_argument(
        "--prefer-paired-normal",
        action="store_true",
        help="When duplicates exist, preserve one normal and one tumor max per patient for paired analyses.",
    )
    parser.add_argument(
        "--keep-all-samples",
        action="store_true",
        help="Export both all matched samples and the primary representative-sample cohort.",
    )
    return parser.parse_args()



def _load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_column_names(df)



def _pick_expression_identifier_column(df: pd.DataFrame, id_column: str | None) -> str:
    if id_column and id_column in df.columns:
        return id_column
    for candidate in DEFAULT_ID_CANDIDATES:
        if candidate in df.columns:
            return candidate
    if is_unnamed_column(df.columns[0]) or looks_like_identifier_series(df.iloc[:, 0]):
        return str(df.columns[0])
    raise KeyError("Unable to determine identifier column for expression matrix.")



def _extract_gene_columns(df: pd.DataFrame, id_column: str) -> list[str]:
    return [column for column in df.columns if column != id_column and not column.startswith("_")]



def _build_exclusion_row(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    result = frame.copy()
    result["exclusion_reason"] = reason
    result["representative_selected"] = False
    return result



def main() -> None:
    """Run cohort curation, grouped fold generation, and aligned export creation."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    counts_df = maybe_transpose_expression_table(_load_csv(args.counts))
    normalized_df = maybe_transpose_expression_table(_load_csv(args.normalized))
    metadata_df = standardize_metadata_columns(_load_csv(args.metadata))

    detected_id_column = detect_identifier((counts_df, normalized_df, metadata_df), args.id_column)
    counts_id_column = _pick_expression_identifier_column(counts_df, detected_id_column)
    normalized_id_column = _pick_expression_identifier_column(normalized_df, detected_id_column)
    metadata_id_column = detected_id_column

    counts_df = add_harmonized_identifiers(counts_df, counts_id_column)
    normalized_df = add_harmonized_identifiers(normalized_df, normalized_id_column)
    metadata_df = add_harmonized_identifiers(metadata_df, metadata_id_column)

    counts_sample_ids = counts_df["_harmonized_sample_id"].fillna(counts_df["_harmonized_id"])
    normalized_sample_ids = normalized_df["_harmonized_sample_id"].fillna(normalized_df["_harmonized_id"])
    metadata_sample_ids = metadata_df["_harmonized_sample_id"].fillna(metadata_df["_harmonized_id"])

    matched_sample_ids = sorted(set(counts_sample_ids.dropna()) & set(normalized_sample_ids.dropna()) & set(metadata_sample_ids.dropna()))
    if not matched_sample_ids:
        raise ValueError("No aligned TCGA sample identifiers were shared across counts, normalized, and metadata tables.")

    master_df = metadata_df[metadata_sample_ids.isin(matched_sample_ids)].copy()
    master_df = infer_sample_type(master_df, args.tumor_code, args.normal_code)
    master_df = build_primary_label(master_df, args.tumor_code, args.normal_code)
    master_df = build_confounder_table(master_df, args.max_missing_confounder_fraction)
    master_df["representative_selected"] = False
    master_df["exclusion_reason"] = pd.NA

    exclusion_logs: list[pd.DataFrame] = []
    unmatched_metadata = metadata_df.loc[~metadata_sample_ids.isin(matched_sample_ids)].copy()
    if not unmatched_metadata.empty:
        exclusion_logs.append(_build_exclusion_row(unmatched_metadata, "unmatched_across_input_tables"))

    unsupported_df = master_df.loc[~master_df["primary_supported_sample"]].copy()
    if not unsupported_df.empty:
        exclusion_logs.append(_build_exclusion_row(unsupported_df, "unsupported_sample_type_for_primary_cohort"))

    missing_patient_df = master_df.loc[master_df["patient_id"].isna()].copy()
    if not missing_patient_df.empty:
        raise ValueError("Failed to derive patient identifiers for one or more matched samples.")

    primary_candidates = master_df.loc[master_df["primary_supported_sample"]].copy()
    primary_df, duplicate_exclusions, duplicate_stats = select_representative_samples(
        primary_candidates,
        prefer_paired_normal=args.prefer_paired_normal,
    )
    exclusion_logs.append(duplicate_exclusions)

    if primary_df.empty:
        raise ValueError("Primary cohort is empty after applying tumor/normal inclusion rules.")

    if primary_df["patient_id"].duplicated().any() and primary_df.groupby("patient_id")["disease_label"].nunique().gt(1).any():
        raise ValueError(
            "Primary cohort contains paired tumor-normal samples for at least one patient. "
            "This is preserved for cohort export, but grouped_outer_folds.csv requires one label per patient. "
            "Re-run without paired retention for fold generation."
        )

    fold_df = create_grouped_folds(primary_df, args.outer_folds, args.random_state)
    primary_df = primary_df.merge(fold_df[["sample_id", "outer_fold"]], on="sample_id", how="left")

    selected_sample_ids = primary_df["sample_id"].astype(str).tolist()
    normalized_primary = align_expression_to_cohort(normalized_df, selected_sample_ids, normalized_id_column)
    counts_primary = align_expression_to_cohort(counts_df, selected_sample_ids, counts_id_column)

    if set(normalized_primary["sample_id"]) != set(selected_sample_ids):
        raise RuntimeError("Normalized matrix alignment did not preserve the curated sample IDs.")
    if set(counts_primary["sample_id"]) != set(selected_sample_ids):
        raise RuntimeError("Counts matrix alignment did not preserve the curated sample IDs.")

    all_columns = [
        "original_id",
        "sample_id",
        "patient_id",
        "sample_type_code",
        "sample_type_label",
        "disease_label",
        "disease_label_name",
        "histology_raw",
        "histology_clean",
        "sex_raw",
        "env_sex",
        "smoking_raw",
        "env_smoking",
        "stage_raw",
        "env_stage",
        "country_raw",
        "env_country_or_region",
        "race_raw",
        "age_at_diagnosis",
        "pack_years_raw",
        "alcohol_history_raw",
        "survival_status_raw",
        "days_to_death",
        "days_to_last_follow_up",
        "metadata_completeness_score",
        "confounder_missing_fraction",
        "confounder_missing_excess",
        "sample_type_conflict",
        "representative_selected",
        "exclusion_reason",
        "outer_fold",
    ]
    for column in all_columns:
        if column not in master_df.columns:
            master_df[column] = pd.NA
        if column not in primary_df.columns:
            primary_df[column] = pd.NA
    master_all_df = master_df[all_columns].sort_values(["patient_id", "sample_id"], kind="mergesort").reset_index(drop=True)
    primary_df = primary_df[all_columns].sort_values(["patient_id", "disease_label", "sample_id"], kind="mergesort").reset_index(drop=True)

    exclusion_log_df = pd.concat([frame for frame in exclusion_logs if not frame.empty], ignore_index=True) if exclusion_logs else pd.DataFrame()
    if not exclusion_log_df.empty:
        for column in all_columns:
            if column not in exclusion_log_df.columns:
                exclusion_log_df[column] = pd.NA
        exclusion_log_df = exclusion_log_df[all_columns]

    summary_payload = {
        "matched_sample_counts": {
            "matched_across_all_inputs": int(len(matched_sample_ids)),
            "matched_normalized_samples": int(normalized_sample_ids.isin(matched_sample_ids).sum()),
            "matched_count_samples": int(counts_sample_ids.isin(matched_sample_ids).sum()),
            "matched_metadata_samples": int(metadata_sample_ids.isin(matched_sample_ids).sum()),
        },
        "tumor_count": int(primary_df["disease_label_name"].eq("tumor").sum()),
        "normal_count": int(primary_df["disease_label_name"].eq("normal").sum()),
        "unique_patient_count": int(primary_df["patient_id"].nunique()),
        "duplicate_patient_count": int(duplicate_stats["duplicate_patient_count"]),
        "excluded_sample_counts_by_reason": {
            str(key): int(value)
            for key, value in exclusion_log_df["exclusion_reason"].value_counts(dropna=False).items()
        }
        if not exclusion_log_df.empty
        else {},
        "available_confounder_counts": _summarize_available_confounders(master_all_df),
        "environment_category_counts": _environment_category_counts(primary_df),
        "fold_distribution": {
            str(fold): {str(label): int(count) for label, count in subgroup.items()}
            for fold, subgroup in fold_df.groupby("outer_fold")["disease_label"].value_counts().unstack(fill_value=0).to_dict(orient="index").items()
        },
    }

    write_outputs(
        output_dir=output_dir,
        master_all_df=master_all_df,
        primary_df=primary_df,
        normalized_primary=normalized_primary,
        counts_primary=counts_primary,
        fold_df=fold_df,
        exclusion_log_df=exclusion_log_df,
        summary_payload=summary_payload,
    )

    print(f"Matched normalized samples: {summary_payload['matched_sample_counts']['matched_normalized_samples']}")
    print(f"Matched count samples: {summary_payload['matched_sample_counts']['matched_count_samples']}")
    print(f"Matched metadata samples: {summary_payload['matched_sample_counts']['matched_metadata_samples']}")
    print(f"Primary retained tumor samples: {summary_payload['tumor_count']}")
    print(f"Primary retained normal samples: {summary_payload['normal_count']}")
    print(f"Unique patients: {summary_payload['unique_patient_count']}")
    print(f"Duplicate-sample reductions: {duplicate_stats['duplicate_sample_reductions']}")
    print(f"Fold distribution: {summary_payload['fold_distribution']}")


if __name__ == "__main__":
    main()
