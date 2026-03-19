#!/usr/bin/env python3
"""Grouped nested cross-validation baselines for TCGA ESCA tumor-vs-normal modeling.

This script consumes the curated Step 2 primary cohort, normalized expression matrix,
and patient-grouped outer folds to benchmark leakage-safe baseline classifiers.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from publication_style import (
    apply_text_style,
    configure_publication_plotting,
    get_nature_palette,
    save_publication_figure,
)

try:  # Optional dependency.
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    XGBClassifier = None

try:  # Optional dependency.
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    LGBMClassifier = None

SUBGROUP_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sex": ("env_sex", "sex_raw", "sex"),
    "smoking": ("env_smoking", "smoking_raw", "smoking_status"),
    "histology": ("env_histology", "histology_clean", "histology_raw", "histology"),
    "stage": ("env_stage", "stage_raw", "stage"),
    "country_or_region": ("env_country_or_region", "country_raw", "country_or_region"),
}
MODEL_COLOR_KEYS = {
    "LR_L1": "normal",
    "LR_ElasticNet": "male",
    "RandomForest": "smoker",
    "XGBoost": "tumor",
    "LightGBM": "escc",
    "HistGB": "eac",
    "MLP": "female",
}


@dataclass(frozen=True)
class RunConfig:
    cohort: str
    normalized: str
    folds: str
    output_dir: str
    label_column: str
    patient_column: str
    sample_column: str
    outer_fold_column: str
    random_seeds: list[int]
    inner_folds: int
    n_jobs: int
    top_variable_genes: int
    min_genes: int
    max_genes: int
    run_mlp: bool
    run_xgboost: bool
    use_lightgbm_if_available: bool
    save_fold_models: bool
    calibration_bins: int


@dataclass(frozen=True)
class PreprocessorArtifacts:
    numeric_columns: list[str]
    retained_columns: list[str]
    selected_columns: list[str]
    variance_threshold: VarianceThreshold
    imputer: SimpleImputer
    scaler: StandardScaler


class GradientBoostingProbabilityAdapter(BaseEstimator, ClassifierMixin):
    """Wrap HistGradientBoostingClassifier with a stable interface for tuning."""

    def __init__(self, learning_rate: float = 0.05, max_depth: int = 3, max_iter: int = 200, random_state: int = 42):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingProbabilityAdapter":
        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.model_.predict_proba(X)
        if probabilities.shape[1] == 1:
            positive = probabilities[:, 0]
            return np.column_stack([1.0 - positive, positive])
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


class XGBoostWrapper(XGBClassifier if XGBClassifier is not None else BaseEstimator):
    """Thin wrapper to expose a clean estimator name when XGBoost is available."""


class LightGBMWrapper(LGBMClassifier if LGBMClassifier is not None else BaseEstimator):
    """Thin wrapper to expose a clean estimator name when LightGBM is available."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, help="Path to master_samples_primary.csv.")
    parser.add_argument("--normalized", required=True, help="Path to normalized_primary_matrix.csv.")
    parser.add_argument("--folds", required=True, help="Path to grouped_outer_folds.csv.")
    parser.add_argument("--output-dir", required=True, help="Output directory for Step 3 results.")
    parser.add_argument("--label-column", default="disease_label", help="Explicit tumor-vs-normal label column.")
    parser.add_argument("--patient-column", default="patient_id", help="Patient grouping column.")
    parser.add_argument("--sample-column", default="sample_id", help="Sample identifier column.")
    parser.add_argument("--outer-fold-column", default="outer_fold", help="Grouped outer fold column.")
    parser.add_argument("--random-seeds", default="42,52,62,72,82", help="Comma-separated deterministic seeds.")
    parser.add_argument("--inner-folds", type=int, default=3, help="Number of grouped inner CV folds.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Reserved parallelism hint for compatible estimators.")
    parser.add_argument("--top-variable-genes", type=int, default=5000, help="Default training-only variable gene cap.")
    parser.add_argument("--min-genes", type=int, default=1000, help="Minimum number of genes to retain if available.")
    parser.add_argument("--max-genes", type=int, default=8000, help="Maximum number of genes to retain after filtering.")
    parser.add_argument("--run-mlp", action="store_true", help="Include a compact MLP baseline.")
    parser.add_argument("--run-xgboost", action="store_true", help="Prefer XGBoost if installed.")
    parser.add_argument("--use-lightgbm-if-available", action="store_true", help="Prefer LightGBM over HistGB when XGBoost is unavailable.")
    parser.add_argument("--save-fold-models", action="store_true", help="Persist fitted outer-fold models as pickle artifacts.")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of bins for calibration summaries.")
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    seeds = [int(item.strip()) for item in str(args.random_seeds).split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one random seed is required.")
    return RunConfig(
        cohort=args.cohort,
        normalized=args.normalized,
        folds=args.folds,
        output_dir=args.output_dir,
        label_column=args.label_column,
        patient_column=args.patient_column,
        sample_column=args.sample_column,
        outer_fold_column=args.outer_fold_column,
        random_seeds=seeds,
        inner_folds=args.inner_folds,
        n_jobs=args.n_jobs,
        top_variable_genes=args.top_variable_genes,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        run_mlp=bool(args.run_mlp),
        run_xgboost=bool(args.run_xgboost),
        use_lightgbm_if_available=bool(args.use_lightgbm_if_available),
        save_fold_models=bool(args.save_fold_models),
        calibration_bins=args.calibration_bins,
    )


def validate_required_columns(df: pd.DataFrame, required: Sequence[str], frame_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")


def load_inputs(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_df = pd.read_csv(config.cohort)
    normalized_df = pd.read_csv(config.normalized)
    folds_df = pd.read_csv(config.folds)
    validate_required_columns(
        cohort_df,
        [config.sample_column, config.patient_column, config.label_column, "disease_label_name"],
        "cohort",
    )
    validate_required_columns(normalized_df, [config.sample_column], "normalized matrix")
    validate_required_columns(
        folds_df,
        [config.sample_column, config.patient_column, config.label_column, config.outer_fold_column],
        "fold table",
    )
    return cohort_df, normalized_df, folds_df


def align_expression_and_cohort(
    cohort_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    sample_column: str,
    outer_fold_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = cohort_df.copy()
    normalized = normalized_df.copy()
    folds = folds_df.copy()

    for frame in (cohort, normalized, folds):
        frame[sample_column] = frame[sample_column].astype(str)

    if cohort[sample_column].duplicated().any():
        raise ValueError("Curated cohort contains duplicate sample IDs.")
    if normalized[sample_column].duplicated().any():
        raise ValueError("Normalized matrix contains duplicate sample IDs.")
    if folds[sample_column].duplicated().any():
        raise ValueError("Fold file contains duplicate sample IDs.")

    shared_sample_ids = sorted(set(cohort[sample_column]) & set(normalized[sample_column]) & set(folds[sample_column]))
    if not shared_sample_ids:
        raise ValueError("No shared sample IDs were found across cohort, normalized matrix, and fold file.")

    cohort = cohort[cohort[sample_column].isin(shared_sample_ids)].copy()
    normalized = normalized[normalized[sample_column].isin(shared_sample_ids)].copy()
    folds = folds[folds[sample_column].isin(shared_sample_ids)].copy()

    aligned = cohort.merge(
        folds[[sample_column, outer_fold_column]] if outer_fold_column in folds.columns else folds[[sample_column]],
        on=sample_column,
        how="inner",
        suffixes=("", "_fold"),
    )
    folded_name = f"{outer_fold_column}_fold"
    if folded_name in aligned.columns and outer_fold_column not in aligned.columns:
        aligned = aligned.rename(columns={folded_name: outer_fold_column})
    for column in folds.columns:
        if column not in aligned.columns and column != sample_column:
            aligned = aligned.merge(folds[[sample_column, column]], on=sample_column, how="left")

    aligned = aligned.sort_values(["outer_fold", sample_column], kind="mergesort").reset_index(drop=True)
    normalized = normalized.set_index(sample_column).loc[aligned[sample_column]].reset_index()
    return aligned, normalized


def validate_grouped_folds(df: pd.DataFrame, label_column: str, patient_column: str, outer_fold_column: str) -> list[int]:
    if df[outer_fold_column].isna().any():
        raise ValueError("Outer fold assignments contain missing values.")
    if not set(df[label_column].dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary with tumor=1 and normal=0.")
    patient_labels = df.groupby(patient_column)[label_column].nunique()
    if (patient_labels > 1).any():
        leaking_patients = patient_labels[patient_labels > 1].index.tolist()[:10]
        raise ValueError(f"Each patient must map to a single label for grouped CV. Problem patients: {leaking_patients}")
    unique_folds = sorted(df[outer_fold_column].astype(int).unique().tolist())
    if len(unique_folds) < 2:
        raise ValueError("At least two outer folds are required.")
    for fold in unique_folds:
        test_patients = set(df.loc[df[outer_fold_column] == fold, patient_column].astype(str))
        train_patients = set(df.loc[df[outer_fold_column] != fold, patient_column].astype(str))
        overlap = test_patients & train_patients
        if overlap:
            raise ValueError(f"Patient leakage detected in outer fold {fold}: {sorted(list(overlap))[:10]}")
    return unique_folds


def build_outer_split(df: pd.DataFrame, outer_fold: int, outer_fold_column: str) -> tuple[np.ndarray, np.ndarray]:
    test_mask = df[outer_fold_column].astype(int).eq(int(outer_fold)).to_numpy()
    train_index = np.flatnonzero(~test_mask)
    test_index = np.flatnonzero(test_mask)
    if len(train_index) == 0 or len(test_index) == 0:
        raise ValueError(f"Outer fold {outer_fold} produced an empty train or test split.")
    return train_index, test_index


def build_inner_grouped_cv(
    cohort_train_df: pd.DataFrame,
    label_column: str,
    patient_column: str,
    inner_folds: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = cohort_train_df[[patient_column, label_column]].drop_duplicates()
    n_groups = len(unique_groups)
    class_counts = unique_groups[label_column].value_counts().to_dict()
    feasible_folds = min(inner_folds, n_groups, min(class_counts.values()))
    if feasible_folds < 2:
        raise ValueError("Grouped inner CV requires at least two patient groups per class in the training split.")
    splitter = StratifiedGroupKFold(n_splits=feasible_folds, shuffle=True, random_state=random_seed)
    splits = list(
        splitter.split(
            cohort_train_df[[patient_column]],
            cohort_train_df[label_column].astype(int),
            groups=cohort_train_df[patient_column].astype(str),
        )
    )
    for train_index, validation_index in splits:
        train_patients = set(cohort_train_df.iloc[train_index][patient_column].astype(str))
        validation_patients = set(cohort_train_df.iloc[validation_index][patient_column].astype(str))
        overlap = train_patients & validation_patients
        if overlap:
            raise ValueError(f"Patient leakage detected in inner CV: {sorted(list(overlap))[:10]}")
    return splits


def _safe_numeric_expression(expr_df: pd.DataFrame, sample_column: str) -> pd.DataFrame:
    working = expr_df.copy()
    non_sample_columns = [column for column in working.columns if column != sample_column]
    for column in non_sample_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    numeric_columns = [column for column in non_sample_columns if pd.api.types.is_numeric_dtype(working[column])]
    if not numeric_columns:
        raise ValueError("Normalized matrix did not contain numeric gene expression columns.")
    return working[[sample_column] + numeric_columns]


def select_training_genes(
    X_train_df: pd.DataFrame,
    sample_column: str,
    top_variable_genes: int,
    min_genes: int,
    max_genes: int,
) -> tuple[list[str], list[str]]:
    numeric_df = _safe_numeric_expression(X_train_df, sample_column)
    numeric_columns = [column for column in numeric_df.columns if column != sample_column]
    variance = numeric_df[numeric_columns].var(axis=0, skipna=True)
    retained_columns = variance[variance > 1e-12].sort_values(ascending=False).index.tolist()
    if not retained_columns:
        raise ValueError("No non-zero-variance genes remained after training-only filtering.")
    target_gene_count = min(max(top_variable_genes, min_genes), max_genes, len(retained_columns))
    if len(retained_columns) < min_genes:
        target_gene_count = len(retained_columns)
    selected_columns = retained_columns[:target_gene_count]
    return retained_columns, selected_columns


def fit_preprocessor(
    X_train_df: pd.DataFrame,
    sample_column: str,
    selected_columns: Sequence[str],
) -> tuple[np.ndarray, PreprocessorArtifacts]:
    numeric_columns = list(selected_columns)
    X_train = X_train_df[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    variance_threshold = VarianceThreshold(threshold=0.0)
    X_train = variance_threshold.fit_transform(X_train)
    retained_mask = variance_threshold.get_support()
    retained_columns = [column for column, keep in zip(numeric_columns, retained_mask) if keep]
    if not retained_columns:
        raise ValueError("Zero-variance filtering removed every selected gene.")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    artifacts = PreprocessorArtifacts(
        numeric_columns=numeric_columns,
        retained_columns=retained_columns,
        selected_columns=numeric_columns,
        variance_threshold=variance_threshold,
        imputer=imputer,
        scaler=scaler,
    )
    return X_train, artifacts


def transform_with_preprocessor(X_df: pd.DataFrame, artifacts: PreprocessorArtifacts) -> np.ndarray:
    X = X_df[artifacts.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = artifacts.variance_threshold.transform(X)
    X = artifacts.imputer.transform(X)
    X = artifacts.scaler.transform(X)
    return X


def get_model_search_spaces(config: RunConfig, seed: int) -> dict[str, tuple[BaseEstimator, list[dict[str, Any]]]]:
    models: dict[str, tuple[BaseEstimator, list[dict[str, Any]]]] = {
        "LR_L1": (
            LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=4000,
                random_state=seed,
                n_jobs=config.n_jobs,
            ),
            list(ParameterGrid({"C": [0.01, 0.1, 1.0, 5.0], "class_weight": [None, "balanced"]})),
        ),
        "LR_ElasticNet": (
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=4000,
                random_state=seed,
                n_jobs=config.n_jobs,
            ),
            list(
                ParameterGrid(
                    {"C": [0.01, 0.1, 1.0, 5.0], "l1_ratio": [0.2, 0.5, 0.8], "class_weight": [None, "balanced"]}
                )
            ),
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=seed, n_jobs=config.n_jobs),
            list(
                ParameterGrid(
                    {
                        "n_estimators": [300, 500],
                        "max_depth": [None, 8, 16],
                        "min_samples_split": [2, 5],
                        "class_weight": [None, "balanced"],
                    }
                )
            ),
        ),
    }
    if config.run_xgboost and XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_estimators=300,
                n_jobs=config.n_jobs,
                use_label_encoder=False,
                verbosity=0,
            ),
            list(
                ParameterGrid(
                    {
                        "learning_rate": [0.03, 0.1],
                        "max_depth": [3, 5],
                        "n_estimators": [200, 400],
                        "subsample": [0.8, 1.0],
                    }
                )
            ),
        )
    elif config.use_lightgbm_if_available and LGBMClassifier is not None:
        models["LightGBM"] = (
            LGBMClassifier(objective="binary", random_state=seed, n_estimators=300, n_jobs=config.n_jobs, verbose=-1),
            list(
                ParameterGrid(
                    {
                        "learning_rate": [0.03, 0.1],
                        "max_depth": [-1, 5, 8],
                        "n_estimators": [200, 400],
                        "subsample": [0.8, 1.0],
                    }
                )
            ),
        )
    else:
        models["HistGB"] = (
            GradientBoostingProbabilityAdapter(random_state=seed),
            list(ParameterGrid({"learning_rate": [0.03, 0.1], "max_depth": [3, 5], "max_iter": [200, 400]})),
        )
    if config.run_mlp:
        models["MLP"] = (
            MLPClassifier(
                random_state=seed,
                early_stopping=True,
                max_iter=400,
                validation_fraction=0.15,
            ),
            list(
                ParameterGrid(
                    {
                        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                        "alpha": [1e-4, 1e-3, 1e-2],
                        "learning_rate_init": [1e-3, 5e-4],
                    }
                )
            ),
        )
    return models


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_model(
    model_name: str,
    estimator: BaseEstimator,
    param_grid: Sequence[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[BaseEstimator, dict[str, Any], pd.DataFrame]:
    search_logs: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    best_estimator: BaseEstimator | None = None

    for index, params in enumerate(param_grid, start=1):
        fold_scores: list[float] = []
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            train_groups = set(groups[inner_train_idx].astype(str))
            valid_groups = set(groups[inner_valid_idx].astype(str))
            overlap = train_groups & valid_groups
            if overlap:
                raise ValueError(f"Patient leakage detected while tuning {model_name}: {sorted(list(overlap))[:10]}")
            candidate = clone(estimator)
            candidate.set_params(**params)
            candidate.fit(X_train[inner_train_idx], y_train[inner_train_idx])
            y_valid_prob = candidate.predict_proba(X_train[inner_valid_idx])[:, 1]
            if len(np.unique(y_train[inner_valid_idx])) < 2:
                score = 0.5
            else:
                score = roc_auc_score(y_train[inner_valid_idx], y_valid_prob)
            fold_scores.append(float(score))
            search_logs.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "candidate_index": index,
                    "inner_fold": inner_fold,
                    "inner_auc": float(score),
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)
            best_estimator = clone(estimator).set_params(**params)

    if best_estimator is None or best_params is None:
        raise RuntimeError(f"Hyperparameter tuning failed for {model_name}.")

    best_estimator.fit(X_train, y_train)
    search_log_df = pd.DataFrame(search_logs)
    search_log_df["mean_inner_auc"] = search_log_df.groupby(["seed", "model_name", "candidate_index"])["inner_auc"].transform("mean")
    search_log_df["selected_best"] = search_log_df["params_json"].eq(json.dumps(best_params, sort_keys=True))
    return best_estimator, best_params, search_log_df


def compute_calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    seed: int,
    outer_fold: int | str,
    n_bins: int,
) -> pd.DataFrame:
    calibration = pd.DataFrame({"y_true": y_true.astype(int), "y_prob": y_prob.astype(float)})
    calibration["bin"] = pd.qcut(calibration["y_prob"], q=min(n_bins, calibration["y_prob"].nunique()), duplicates="drop")
    summary = (
        calibration.groupby("bin", observed=False)
        .agg(mean_predicted_probability=("y_prob", "mean"), observed_fraction=("y_true", "mean"), samples=("y_true", "size"))
        .reset_index(drop=True)
    )
    summary["model_name"] = model_name
    summary["seed"] = seed
    summary["outer_fold"] = outer_fold
    return summary


def fit_and_predict_outer_fold(
    cohort_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    config: RunConfig,
    outer_fold: int,
    seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_index, test_index = build_outer_split(cohort_df, outer_fold, config.outer_fold_column)
    cohort_train_df = cohort_df.iloc[train_index].reset_index(drop=True)
    cohort_test_df = cohort_df.iloc[test_index].reset_index(drop=True)
    expr_train_df = expression_df.iloc[train_index].reset_index(drop=True)
    expr_test_df = expression_df.iloc[test_index].reset_index(drop=True)

    train_patients = set(cohort_train_df[config.patient_column].astype(str))
    test_patients = set(cohort_test_df[config.patient_column].astype(str))
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(f"Outer fold leakage detected for fold {outer_fold}: {sorted(list(overlap))[:10]}")

    retained_columns, selected_columns = select_training_genes(
        expr_train_df,
        config.sample_column,
        config.top_variable_genes,
        config.min_genes,
        config.max_genes,
    )
    X_train, preprocessor = fit_preprocessor(expr_train_df, config.sample_column, selected_columns)
    X_test = transform_with_preprocessor(expr_test_df, preprocessor)
    y_train = cohort_train_df[config.label_column].astype(int).to_numpy()
    y_test = cohort_test_df[config.label_column].astype(int).to_numpy()
    inner_splits = build_inner_grouped_cv(
        cohort_train_df,
        config.label_column,
        config.patient_column,
        config.inner_folds,
        seed,
    )
    groups = cohort_train_df[config.patient_column].astype(str).to_numpy()

    fold_metric_rows: list[dict[str, Any]] = []
    fold_prediction_rows: list[pd.DataFrame] = []
    selection_logs: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    feature_space_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for model_name, (estimator, param_grid) in get_model_search_spaces(config, seed).items():
        best_estimator, best_params, search_log_df = tune_model(
            model_name,
            estimator,
            param_grid,
            X_train,
            y_train,
            groups,
            inner_splits,
            seed,
        )
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_binary_metrics(y_test, y_prob)
        metrics.update(
            {
                "model_name": model_name,
                "seed": seed,
                "outer_fold": outer_fold,
                "n_train_samples": int(len(train_index)),
                "n_test_samples": int(len(test_index)),
                "n_train_patients": int(cohort_train_df[config.patient_column].nunique()),
                "n_test_patients": int(cohort_test_df[config.patient_column].nunique()),
                "best_params_json": json.dumps(best_params, sort_keys=True),
            }
        )
        fold_metric_rows.append(metrics)
        selection_logs.append(search_log_df.assign(outer_fold=outer_fold, best_params_json=json.dumps(best_params, sort_keys=True)))
        fold_prediction_rows.append(
            cohort_test_df[[config.sample_column, config.patient_column, config.label_column, config.outer_fold_column]]
            .copy()
            .assign(
                seed=seed,
                model_name=model_name,
                y_true=y_test,
                y_prob=y_prob,
                y_pred=y_pred,
            )
        )
        calibration_rows.append(compute_calibration_table(y_test, y_prob, model_name, seed, outer_fold, config.calibration_bins))
        feature_space_rows.append(
            {
                "seed": seed,
                "outer_fold": outer_fold,
                "model_name": model_name,
                "original_gene_count": int(len([column for column in expr_train_df.columns if column != config.sample_column])),
                "retained_gene_count": int(len(retained_columns)),
                "selected_variable_genes": int(len(preprocessor.retained_columns)),
            }
        )
        fitted_models[model_name] = {"estimator": best_estimator, "preprocessor": preprocessor, "selected_genes": preprocessor.retained_columns}

    if config.save_fold_models:
        model_dir = output_dir / "saved_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = model_dir / f"seed_{seed}_outer_fold_{outer_fold}.pkl"
        with artifact_path.open("wb") as handle:
            pickle.dump(fitted_models, handle)

    return (
        pd.DataFrame(fold_metric_rows),
        pd.concat(fold_prediction_rows, ignore_index=True),
        pd.concat(selection_logs, ignore_index=True),
        pd.concat(calibration_rows, ignore_index=True),
        {"feature_space": pd.DataFrame(feature_space_rows)},
    )


def compute_subgroup_metrics(
    prediction_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    sample_column: str,
    min_samples: int = 10,
    min_class_count: int = 3,
) -> pd.DataFrame:
    merged = prediction_df.merge(cohort_df, on=sample_column, how="left", suffixes=("", "_cohort"))
    rows: list[dict[str, Any]] = []
    for subgroup_name, candidates in SUBGROUP_COLUMN_CANDIDATES.items():
        column = next((candidate for candidate in candidates if candidate in merged.columns), None)
        if column is None:
            continue
        for value, subgroup_df in merged.groupby(column, dropna=False):
            label_counts = subgroup_df["y_true"].value_counts()
            if len(subgroup_df) < min_samples or label_counts.min() < min_class_count or len(label_counts) < 2:
                rows.append(
                    {
                        "model_name": subgroup_df["model_name"].iloc[0],
                        "seed": subgroup_df["seed"].iloc[0],
                        "subgroup_name": subgroup_name,
                        "subgroup_value": "unknown" if pd.isna(value) else str(value),
                        "n_samples": int(len(subgroup_df)),
                        "status": "too_small",
                    }
                )
                continue
            metrics = compute_binary_metrics(subgroup_df["y_true"].to_numpy(), subgroup_df["y_prob"].to_numpy())
            metrics.update(
                {
                    "model_name": subgroup_df["model_name"].iloc[0],
                    "seed": subgroup_df["seed"].iloc[0],
                    "subgroup_name": subgroup_name,
                    "subgroup_value": "unknown" if pd.isna(value) else str(value),
                    "n_samples": int(len(subgroup_df)),
                    "status": "ok",
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def aggregate_results(fold_metrics_df: pd.DataFrame, oof_predictions_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "auprc",
        "balanced_accuracy",
        "mcc",
        "f1",
        "sensitivity",
        "specificity",
        "brier_score",
    ]
    summary = (
        fold_metrics_df.groupby("model_name")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model_name" if column == ("model_name", "") else "_".join([part for part in column if part]) for column in summary.columns]

    pooled_rows: list[dict[str, Any]] = []
    for model_name, group in oof_predictions_df.groupby("model_name"):
        pooled_metrics = compute_binary_metrics(group["y_true"].to_numpy(), group["y_prob"].to_numpy())
        pooled_rows.append({"model_name": model_name, **{f"pooled_{key}": value for key, value in pooled_metrics.items()}})
    return summary.merge(pd.DataFrame(pooled_rows), on="model_name", how="left")


def _model_colors(model_names: Iterable[str]) -> dict[str, str]:
    palette = get_nature_palette()
    return {name: palette.get(MODEL_COLOR_KEYS.get(name, "unknown"), palette["unknown"]) for name in model_names}


def plot_roc_comparison(oof_predictions_df: pd.DataFrame, figures_dir: Path) -> None:
    configure_publication_plotting()
    model_colors = _model_colors(oof_predictions_df["model_name"].unique())
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    for model_name, group in sorted(oof_predictions_df.groupby("model_name"), key=lambda item: item[0]):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_prob"])
        auc_value = roc_auc_score(group["y_true"], group["y_prob"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUROC={auc_value:.3f})", color=model_colors[model_name])
    ax.plot([0, 1], [0, 1], linestyle="--", color=get_nature_palette()["unknown"], linewidth=1.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Baseline ROC comparison")
    ax.legend(loc="lower right", title="Model")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_baseline_roc.svg")
    plt.close(fig)


def plot_pr_comparison(oof_predictions_df: pd.DataFrame, figures_dir: Path) -> None:
    configure_publication_plotting()
    model_colors = _model_colors(oof_predictions_df["model_name"].unique())
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    for model_name, group in sorted(oof_predictions_df.groupby("model_name"), key=lambda item: item[0]):
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_prob"])
        ap_value = average_precision_score(group["y_true"], group["y_prob"])
        ax.plot(recall, precision, label=f"{model_name} (AUPRC={ap_value:.3f})", color=model_colors[model_name])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Baseline precision-recall comparison")
    ax.legend(loc="lower left", title="Model")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_baseline_pr.svg")
    plt.close(fig)


def plot_performance_summary(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    configure_publication_plotting()
    metrics = ["auroc_mean", "auprc_mean", "balanced_accuracy_mean", "mcc_mean"]
    labels = ["AUROC", "AUPRC", "Balanced Acc.", "MCC"]
    x = np.arange(len(summary_df))
    width = 0.18
    model_colors = _model_colors(summary_df["model_name"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(
            x + (idx - 1.5) * width,
            summary_df[metric],
            width=width,
            label=label,
            alpha=0.9,
            color=[model_colors[name] for name in summary_df["model_name"]],
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model_name"], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Grouped baseline performance summary")
    ax.legend(ncol=2, title="Metric")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_baseline_performance_bar.svg")
    plt.close(fig)


def plot_calibration_curves(calibration_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    configure_publication_plotting()
    top_models = summary_df.sort_values(["auroc_mean", "auprc_mean"], ascending=False)["model_name"].head(3).tolist()
    model_colors = _model_colors(top_models)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    for model_name in top_models:
        subset = calibration_df[calibration_df["model_name"] == model_name]
        grouped = subset.groupby("mean_predicted_probability", as_index=False).agg(observed_fraction=("observed_fraction", "mean"))
        ax.plot(
            grouped["mean_predicted_probability"],
            grouped["observed_fraction"],
            marker="o",
            label=model_name,
            color=model_colors[model_name],
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=get_nature_palette()["unknown"], linewidth=1.4)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed tumor fraction")
    ax.set_title("Baseline calibration comparison")
    ax.legend(title="Model")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_calibration.svg")
    plt.close(fig)


def plot_subgroup_summary(subgroup_df: pd.DataFrame, figures_dir: Path) -> None:
    configure_publication_plotting()
    filtered = subgroup_df[(subgroup_df.get("status") == "ok") & subgroup_df["subgroup_name"].isin(["sex", "smoking", "histology", "stage", "country_or_region"])]
    if filtered.empty:
        return
    ranking = filtered.groupby("model_name")["auroc"].mean().sort_values(ascending=False)
    top_models = ranking.head(3).index.tolist()
    filtered = filtered[filtered["model_name"].isin(top_models)].copy()
    filtered["label"] = filtered["subgroup_name"] + ": " + filtered["subgroup_value"]
    filtered = filtered.sort_values(["subgroup_name", "label", "model_name"], kind="mergesort")
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(filtered["label"].unique()) * len(top_models))), constrained_layout=True)
    positions = np.arange(len(filtered["label"].unique()))
    width = 0.22
    labels = filtered["label"].drop_duplicates().tolist()
    model_colors = _model_colors(top_models)
    for idx, model_name in enumerate(top_models):
        model_values = []
        for label in labels:
            subset = filtered[(filtered["label"] == label) & (filtered["model_name"] == model_name)]
            model_values.append(subset["auroc"].mean() if not subset.empty else np.nan)
        ax.bar(positions + (idx - (len(top_models) - 1) / 2) * width, model_values, width=width, label=model_name, color=model_colors[model_name])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Subgroup baseline performance")
    ax.legend(title="Model")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_subgroup_performance.svg")
    plt.close(fig)


def plot_confusion_summary(oof_predictions_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    configure_publication_plotting()
    best_model = summary_df.sort_values(["auroc_mean", "auprc_mean"], ascending=False)["model_name"].iloc[0]
    subset = oof_predictions_df[oof_predictions_df["model_name"] == best_model]
    cm = confusion_matrix(subset["y_true"], subset["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    image = ax.imshow(cm, cmap="Greys")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, f"{int(value)}", ha="center", va="center", fontweight="bold", fontfamily="serif")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal pred.", "Tumor pred."])
    ax.set_yticklabels(["Normal true", "Tumor true"])
    ax.set_title(f"Confusion summary: {best_model}")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_confusion_summary.svg")
    plt.close(fig)


def plot_fold_distribution(cohort_df: pd.DataFrame, outer_fold_column: str, label_column: str, figures_dir: Path) -> None:
    configure_publication_plotting()
    counts = cohort_df.groupby([outer_fold_column, label_column]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    bottom = np.zeros(len(counts), dtype=float)
    palette = get_nature_palette()
    for label_value, color_key, label_name in [(1, "tumor", "Tumor"), (0, "normal", "Normal")]:
        values = counts.get(label_value, pd.Series(0, index=counts.index)).to_numpy(dtype=float)
        ax.bar(counts.index.astype(str), values, bottom=bottom, label=label_name, color=palette[color_key], width=0.68)
        bottom += values
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Samples")
    ax.set_title("Grouped fold distribution")
    ax.legend(title="Class")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure4_fold_distribution.svg")
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    fold_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    oof_predictions_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    selection_log_df: pd.DataFrame,
    feature_space_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    config: RunConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_df.to_csv(output_dir / "baseline_fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "baseline_summary_metrics.csv", index=False)
    oof_predictions_df.to_csv(output_dir / "baseline_oof_predictions.csv", index=False)
    subgroup_df.to_csv(output_dir / "baseline_subgroup_metrics.csv", index=False)
    selection_log_df.to_csv(output_dir / "model_selection_log.csv", index=False)
    feature_space_df.to_csv(output_dir / "baseline_feature_space_summary.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_data.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)


def update_readme_step3(readme_path: Path) -> None:
    readme_text = """# C-A_esophageal

## Project overview

This repository implements a staged TCGA ESCA analysis workflow for tumor-vs-normal modeling. The study design is intentionally sequenced as:

1. **Stage I:** cohort curation and preprocessing.
2. **Stage II:** deep invariant sparse modeling.
3. **Stage III:** candidate driver prioritization and downstream validation.

Step 3 is the bridge between Stage I and Stage II. It establishes a leakage-safe, patient-grouped, manuscript-grade baseline benchmarking framework before the sparse invariant adversarial model is introduced.

## Pipeline overview

1. **Step 1 – generic preprocessing scaffold:** reusable data loading, harmonization, scaling, and exploratory train/test export utilities.
2. **Step 2 – curated primary cohort construction:** explicit tumor-vs-normal label creation, primary-cohort filtering, aligned primary matrices, and grouped outer-fold creation.
3. **Step 3 – grouped nested baseline benchmarking:** grouped nested cross-validation, strong baseline classifiers, pooled out-of-fold predictions, subgroup analyses, and publication-grade figures.
4. **Step 4 – next planned step:** sparse invariant adversarial model training on the curated Step 2 cohort and the grouped evaluation backbone defined in Step 3.

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
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def determine_winners(summary_df: pd.DataFrame) -> tuple[str, str, str]:
    ordered = summary_df.sort_values(
        ["auroc_mean", "auprc_mean", "balanced_accuracy_mean", "mcc_mean", "brier_score_mean"],
        ascending=[False, False, False, False, True],
    )
    best_baseline = str(ordered["model_name"].iloc[0])
    best_calibrated = str(summary_df.sort_values(["brier_score_mean", "auroc_mean"], ascending=[True, False])["model_name"].iloc[0])
    stability_frame = summary_df.assign(stability_score=summary_df["auroc_std"].fillna(np.inf) + summary_df["auprc_std"].fillna(np.inf))
    most_stable = str(stability_frame.sort_values(["stability_score", "auroc_mean"], ascending=[True, False])["model_name"].iloc[0])
    return best_baseline, best_calibrated, most_stable


def main() -> None:
    config = build_run_config(parse_args())
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cohort_df, normalized_df, folds_df = load_inputs(config)
    cohort_df, normalized_df = align_expression_and_cohort(cohort_df, normalized_df, folds_df, config.sample_column, config.outer_fold_column)
    outer_folds = validate_grouped_folds(cohort_df, config.label_column, config.patient_column, config.outer_fold_column)

    fold_metrics_list: list[pd.DataFrame] = []
    oof_predictions_list: list[pd.DataFrame] = []
    selection_logs: list[pd.DataFrame] = []
    calibration_list: list[pd.DataFrame] = []
    feature_space_list: list[pd.DataFrame] = []

    for seed in config.random_seeds:
        print(f"[Step3] Running grouped nested CV for seed={seed} across {len(outer_folds)} outer folds.")
        for outer_fold in outer_folds:
            print(f"[Step3] Seed={seed} outer_fold={outer_fold}: fitting baseline models.")
            fold_metrics_df, oof_df, selection_log_df, calibration_df, extras = fit_and_predict_outer_fold(
                cohort_df,
                normalized_df,
                config,
                outer_fold,
                seed,
                output_dir,
            )
            fold_metrics_list.append(fold_metrics_df)
            oof_predictions_list.append(oof_df)
            selection_logs.append(selection_log_df)
            calibration_list.append(calibration_df)
            feature_space_list.append(extras["feature_space"])

    fold_metrics_df = pd.concat(fold_metrics_list, ignore_index=True).sort_values(["model_name", "seed", "outer_fold"], kind="mergesort")
    oof_predictions_df = pd.concat(oof_predictions_list, ignore_index=True).sort_values(["model_name", "seed", "outer_fold", config.sample_column], kind="mergesort")
    selection_log_df = pd.concat(selection_logs, ignore_index=True).sort_values(["model_name", "seed", "outer_fold", "candidate_index", "inner_fold"], kind="mergesort")
    calibration_df = pd.concat(calibration_list, ignore_index=True).sort_values(["model_name", "seed", "outer_fold", "mean_predicted_probability"], kind="mergesort")
    feature_space_df = pd.concat(feature_space_list, ignore_index=True).sort_values(["model_name", "seed", "outer_fold"], kind="mergesort")
    subgroup_df = (
        oof_predictions_df.groupby(["model_name", "seed"], group_keys=False)
        .apply(lambda frame: compute_subgroup_metrics(frame, cohort_df, config.sample_column))
        .reset_index(drop=True)
    )
    summary_df = aggregate_results(fold_metrics_df, oof_predictions_df)

    save_outputs(
        output_dir,
        fold_metrics_df,
        summary_df,
        oof_predictions_df,
        subgroup_df,
        selection_log_df,
        feature_space_df,
        calibration_df,
        config,
    )

    plot_roc_comparison(oof_predictions_df, figures_dir)
    plot_pr_comparison(oof_predictions_df, figures_dir)
    plot_performance_summary(summary_df, figures_dir)
    plot_calibration_curves(calibration_df, summary_df, figures_dir)
    plot_subgroup_summary(subgroup_df, figures_dir)
    plot_confusion_summary(oof_predictions_df, summary_df, figures_dir)
    plot_fold_distribution(cohort_df, config.outer_fold_column, config.label_column, figures_dir)
    update_readme_step3(Path("README.md"))

    best_baseline, best_calibrated, most_stable = determine_winners(summary_df)
    print(
        "[Step3] Summary | "
        f"best baseline by AUROC: {best_baseline} | "
        f"best calibrated baseline: {best_calibrated} | "
        f"most stable baseline: {most_stable} | "
        f"outer folds: {len(outer_folds)} | "
        f"seeds: {len(config.random_seeds)} | "
        f"samples evaluated: {oof_predictions_df[config.sample_column].nunique()}"
    )


if __name__ == "__main__":
    main()
