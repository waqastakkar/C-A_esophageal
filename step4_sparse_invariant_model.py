#!/usr/bin/env python3
"""Step 4: sparse invariant adversarial autoencoder-classifier for TCGA ESCA."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_model_utils import (
    SparseInvariantAdversarialAutoencoderClassifier,
    choose_hidden_dims,
    encode_label_series,
    environment_risk_variance,
    set_random_seed,
    transform_with_existing_classes,
)
from publication_style import (
    apply_text_style,
    configure_publication_plotting,
    get_nature_palette,
    save_publication_figure,
)

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


@dataclass(frozen=True)
class RunConfig:
    cohort: str
    normalized: str
    folds: str
    baseline_summary: str
    baseline_oof: str
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
    latent_dim: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    classifier_weight: float
    reconstruction_weight: float
    sparsity_weight: float
    adversary_weight: float
    invariance_weight: float
    use_hard_concrete_gate: bool
    use_l1_gate: bool
    environment_columns: list[str]
    confounder_columns: list[str]
    save_fold_models: bool
    save_latent_embeddings: bool
    run_umap: bool
    mixed_precision: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--normalized", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--baseline-summary", required=True)
    p.add_argument("--baseline-oof", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label-column", default="disease_label")
    p.add_argument("--patient-column", default="patient_id")
    p.add_argument("--sample-column", default="sample_id")
    p.add_argument("--outer-fold-column", default="outer_fold")
    p.add_argument("--random-seeds", default="42,52,62,72,82")
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=0)
    p.add_argument("--top-variable-genes", type=int, default=5000)
    p.add_argument("--min-genes", type=int, default=1000)
    p.add_argument("--max-genes", type=int, default=8000)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--classifier-weight", type=float, default=1.0)
    p.add_argument("--reconstruction-weight", type=float, default=1.0)
    p.add_argument("--sparsity-weight", type=float, default=1e-4)
    p.add_argument("--adversary-weight", type=float, default=0.5)
    p.add_argument("--invariance-weight", type=float, default=0.1)
    p.add_argument("--use-hard-concrete-gate", action="store_true")
    p.add_argument("--use-l1-gate", action="store_true")
    p.add_argument("--environment-columns", default="env_sex,env_smoking,env_histology,env_stage,env_country_or_region")
    p.add_argument("--confounder-columns", default="env_sex,env_smoking,env_histology,env_stage,env_country_or_region")
    p.add_argument("--save-fold-models", action="store_true")
    p.add_argument("--save-latent-embeddings", action="store_true")
    p.add_argument("--run-umap", action="store_true")
    p.add_argument("--mixed-precision", action="store_true")
    return p.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    seeds = [int(x.strip()) for x in args.random_seeds.split(",") if x.strip()]
    return RunConfig(
        cohort=args.cohort,
        normalized=args.normalized,
        folds=args.folds,
        baseline_summary=args.baseline_summary,
        baseline_oof=args.baseline_oof,
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
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        classifier_weight=args.classifier_weight,
        reconstruction_weight=args.reconstruction_weight,
        sparsity_weight=args.sparsity_weight,
        adversary_weight=args.adversary_weight,
        invariance_weight=args.invariance_weight,
        use_hard_concrete_gate=bool(args.use_hard_concrete_gate),
        use_l1_gate=bool(args.use_l1_gate) or not bool(args.use_hard_concrete_gate),
        environment_columns=[c.strip() for c in args.environment_columns.split(",") if c.strip()],
        confounder_columns=[c.strip() for c in args.confounder_columns.split(",") if c.strip()],
        save_fold_models=bool(args.save_fold_models),
        save_latent_embeddings=bool(args.save_latent_embeddings),
        run_umap=bool(args.run_umap),
        mixed_precision=bool(args.mixed_precision),
    )


def validate_required_columns(df: pd.DataFrame, required: Sequence[str], frame_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{frame_name} missing required columns: {missing}")


def load_inputs(config: RunConfig):
    cohort = pd.read_csv(config.cohort)
    normalized = pd.read_csv(config.normalized)
    folds = pd.read_csv(config.folds)
    baseline_summary = pd.read_csv(config.baseline_summary)
    baseline_oof = pd.read_csv(config.baseline_oof)
    validate_required_columns(cohort, [config.sample_column, config.patient_column, config.label_column], "cohort")
    validate_required_columns(normalized, [config.sample_column], "normalized")
    validate_required_columns(folds, [config.sample_column, config.patient_column, config.label_column, config.outer_fold_column], "folds")
    return cohort, normalized, folds, baseline_summary, baseline_oof


def align_inputs(cohort_df: pd.DataFrame, normalized_df: pd.DataFrame, folds_df: pd.DataFrame, config: RunConfig):
    for df in (cohort_df, normalized_df, folds_df):
        df[config.sample_column] = df[config.sample_column].astype(str)
    if cohort_df[config.sample_column].duplicated().any() or normalized_df[config.sample_column].duplicated().any() or folds_df[config.sample_column].duplicated().any():
        raise ValueError("Duplicate sample identifiers detected in inputs.")
    shared = sorted(set(cohort_df[config.sample_column]) & set(normalized_df[config.sample_column]) & set(folds_df[config.sample_column]))
    if not shared:
        raise ValueError("No shared sample IDs across cohort, normalized matrix, and folds.")
    cohort = cohort_df[cohort_df[config.sample_column].isin(shared)].copy()
    norm = normalized_df[normalized_df[config.sample_column].isin(shared)].copy()
    folds = folds_df[folds_df[config.sample_column].isin(shared)].copy()
    merged = cohort.merge(folds.drop(columns=[config.label_column], errors="ignore"), on=[config.sample_column, config.patient_column], how="inner")
    merged = merged.sort_values([config.outer_fold_column, config.sample_column], kind="mergesort").reset_index(drop=True)
    norm = norm.set_index(config.sample_column).loc[merged[config.sample_column]].reset_index()
    return merged, norm


def validate_grouped_folds(df: pd.DataFrame, config: RunConfig) -> list[int]:
    patient_labels = df.groupby(config.patient_column)[config.label_column].nunique()
    if (patient_labels > 1).any():
        raise ValueError("Each patient must map to a single disease label.")
    folds = sorted(df[config.outer_fold_column].astype(int).unique().tolist())
    for fold in folds:
        train_patients = set(df.loc[df[config.outer_fold_column] != fold, config.patient_column].astype(str))
        test_patients = set(df.loc[df[config.outer_fold_column] == fold, config.patient_column].astype(str))
        if train_patients & test_patients:
            raise ValueError(f"Patient leakage detected in outer fold {fold}.")
    return folds


def build_outer_split(df: pd.DataFrame, fold: int, outer_fold_column: str):
    test_mask = df[outer_fold_column].astype(int).eq(int(fold)).to_numpy()
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def build_inner_grouped_validation(cohort_train_df: pd.DataFrame, config: RunConfig, seed: int):
    unique_groups = cohort_train_df[[config.patient_column, config.label_column]].drop_duplicates()
    feasible = min(config.inner_folds, len(unique_groups), unique_groups[config.label_column].value_counts().min())
    if feasible < 2:
        raise ValueError("Need at least two patient groups per class for inner grouped validation.")
    cv = StratifiedGroupKFold(n_splits=int(feasible), shuffle=True, random_state=seed)
    return list(cv.split(cohort_train_df, cohort_train_df[config.label_column].astype(int), groups=cohort_train_df[config.patient_column].astype(str)))


def select_training_genes(X_train_df: pd.DataFrame, sample_column: str, top_variable_genes: int, min_genes: int, max_genes: int):
    numeric = X_train_df.drop(columns=[sample_column]).apply(pd.to_numeric, errors="coerce")
    variance = numeric.var(axis=0, skipna=True)
    retained = variance[variance > 1e-12].sort_values(ascending=False).index.tolist()
    if not retained:
        raise ValueError("No informative genes after training-only variance filtering.")
    target = min(max(top_variable_genes, min_genes), max_genes, len(retained))
    if len(retained) < min_genes:
        target = len(retained)
    return retained, retained[:target]


def fit_expression_scaler(X_train_df: pd.DataFrame, selected_columns: Sequence[str]):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(X_train_df[list(selected_columns)].apply(pd.to_numeric, errors="coerce"))
    X_train = scaler.fit_transform(X_train)
    return X_train.astype(np.float32), imputer, scaler


def transform_expression(X_df: pd.DataFrame, selected_columns: Sequence[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    X = imputer.transform(X_df[list(selected_columns)].apply(pd.to_numeric, errors="coerce"))
    return scaler.transform(X).astype(np.float32)


def encode_confounders(cohort_df: pd.DataFrame, columns: Sequence[str]):
    encoded = {}
    skipped = []
    for column in columns:
        if column not in cohort_df.columns:
            skipped.append({"column": column, "reason": "missing"})
            continue
        labels = encode_label_series(cohort_df[column], column)
        if len(labels.classes) < 2:
            skipped.append({"column": column, "reason": "<2 usable levels"})
            continue
        encoded[column] = labels
    return encoded, skipped


def encode_environments(cohort_df: pd.DataFrame, columns: Sequence[str]):
    return encode_confounders(cohort_df, columns)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def build_model(input_dim: int, config: RunConfig, adversary_specs: dict[str, int]) -> SparseInvariantAdversarialAutoencoderClassifier:
    return SparseInvariantAdversarialAutoencoderClassifier(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=choose_hidden_dims(input_dim, config.latent_dim),
        dropout=0.2,
        adversary_specs=adversary_specs,
        grl_lambda=config.adversary_weight,
        use_l1_gate=config.use_l1_gate,
        use_hard_concrete_gate=config.use_hard_concrete_gate,
    )


def adversarial_loss_fn(outputs: dict[str, torch.Tensor], confounder_tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    if not outputs["adversary_outputs"]:
        zero = outputs["logits"].new_tensor(0.0)
        return zero, {}, {}
    criterion = nn.CrossEntropyLoss()
    losses = []
    metrics = {}
    for name, logits in outputs["adversary_outputs"].items():
        targets = confounder_tensors[name]
        valid_mask = targets.ge(0)
        if int(valid_mask.sum().item()) < 2:
            continue
        loss = criterion(logits[valid_mask], targets[valid_mask])
        losses.append(loss)
        pred = logits[valid_mask].argmax(dim=1)
        metrics[name] = float((pred == targets[valid_mask]).float().mean().detach().cpu())
    if not losses:
        zero = outputs["logits"].new_tensor(0.0)
        return zero, metrics, metrics
    return torch.stack(losses).mean(), metrics, metrics


def compute_total_loss(outputs, y, confounder_tensors, env_tensors, config: RunConfig):
    class_weights = None
    pos_frac = y.float().mean().item()
    if 0.0 < pos_frac < 1.0:
        pos_weight = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-6), device=y.device)
        class_weights = pos_weight
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)(outputs["logits"], y.float())
    reconstruction_loss = nn.MSELoss()(outputs["reconstruction"], outputs["gated_x"].detach())
    sparsity_loss = outputs["gate_values"].abs().mean()
    adversarial_loss, adv_metrics, adv_accuracy = adversarial_loss_fn(outputs, confounder_tensors)
    invariance_loss, inv_details = environment_risk_variance(outputs["logits"], y.float(), env_tensors)
    total = (
        config.classifier_weight * classification_loss
        + config.reconstruction_weight * reconstruction_loss
        + config.sparsity_weight * sparsity_loss
        + config.adversary_weight * adversarial_loss
        + config.invariance_weight * invariance_loss
    )
    return total, {
        "classification_loss": float(classification_loss.detach().cpu()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "adversarial_loss": float(adversarial_loss.detach().cpu()),
        "invariance_loss": float(invariance_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "adversary_accuracy_mean": float(np.mean(list(adv_accuracy.values()))) if adv_accuracy else np.nan,
        "environment_penalty_mean": float(np.nanmean(list(inv_details.values()))) if inv_details else np.nan,
    }, adv_metrics, inv_details


def make_dataloader(X, y, confounders, environments, batch_size, shuffle):
    tensors = [torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)]
    names = []
    for name, values in confounders.items():
        tensors.append(torch.tensor(values, dtype=torch.long))
        names.append(("conf", name))
    for name, values in environments.items():
        tensors.append(torch.tensor(values, dtype=torch.long))
        names.append(("env", name))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle), names


def unpack_batch(batch, names, device):
    x = batch[0].to(device)
    y = batch[1].to(device)
    confounders = {}
    environments = {}
    for tensor, (kind, name) in zip(batch[2:], names):
        if kind == "conf":
            confounders[name] = tensor.to(device)
        else:
            environments[name] = tensor.to(device)
    return x, y, confounders, environments


def train_one_epoch(model, loader, names, optimizer, device, config: RunConfig):
    model.train()
    rows = []
    for batch in loader:
        x, y, confounders, environments = unpack_batch(batch, names, device)
        optimizer.zero_grad()
        outputs = model(x)
        total, metrics, _, _ = compute_total_loss(outputs, y, confounders, environments, config)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        rows.append(metrics)
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def evaluate_model(model, loader, names, device, config: RunConfig):
    model.eval()
    all_probs, all_y, rows = [], [], []
    latents = []
    with torch.no_grad():
        for batch in loader:
            x, y, confounders, environments = unpack_batch(batch, names, device)
            outputs = model(x)
            _, metrics, adv_metrics, inv_details = compute_total_loss(outputs, y, confounders, environments, config)
            probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.detach().cpu().numpy())
            latents.append(outputs["z"].detach().cpu().numpy())
            row = dict(metrics)
            row.update({f"adv_acc_{k}": v for k, v in adv_metrics.items()})
            row.update({f"inv_{k}": v for k, v in inv_details.items()})
            rows.append(row)
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_probs)
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics.update(pd.DataFrame(rows).mean(numeric_only=True).to_dict())
    return metrics, y_true, y_prob, np.vstack(latents)


def fit_one_outer_fold(cohort_df, expression_df, config: RunConfig, outer_fold: int, seed: int, output_dir: Path, device: torch.device):
    train_idx, test_idx = build_outer_split(cohort_df, outer_fold, config.outer_fold_column)
    cohort_train = cohort_df.iloc[train_idx].reset_index(drop=True)
    cohort_test = cohort_df.iloc[test_idx].reset_index(drop=True)
    expr_train = expression_df.iloc[train_idx].reset_index(drop=True)
    expr_test = expression_df.iloc[test_idx].reset_index(drop=True)

    retained, selected = select_training_genes(expr_train, config.sample_column, config.top_variable_genes, config.min_genes, config.max_genes)
    X_train, imputer, scaler = fit_expression_scaler(expr_train, selected)
    X_test = transform_expression(expr_test, selected, imputer, scaler)
    y_train = cohort_train[config.label_column].astype(int).to_numpy()
    y_test = cohort_test[config.label_column].astype(int).to_numpy()

    train_confounders, conf_skips = encode_confounders(cohort_train, config.confounder_columns)
    train_envs, env_skips = encode_environments(cohort_train, config.environment_columns)
    test_confounders = {
        name: transform_with_existing_classes(
            cohort_test[name] if name in cohort_test.columns else pd.Series(["Unknown"] * len(cohort_test)),
            name,
            labels.classes,
        ).values
        for name, labels in train_confounders.items()
    }
    test_envs = {
        name: transform_with_existing_classes(
            cohort_test[name] if name in cohort_test.columns else pd.Series(["Unknown"] * len(cohort_test)),
            name,
            labels.classes,
        ).values
        for name, labels in train_envs.items()
    }

    inner_splits = build_inner_grouped_validation(cohort_train, config, seed)
    search_space = list(ParameterGrid({
        "learning_rate": [config.learning_rate, config.learning_rate / 2],
        "reconstruction_weight": [config.reconstruction_weight, max(config.reconstruction_weight / 2, 0.1)],
        "sparsity_weight": [config.sparsity_weight, config.sparsity_weight * 5],
        "adversary_weight": [config.adversary_weight],
        "invariance_weight": [config.invariance_weight],
    }))

    best = None
    selection_log = []
    for candidate_idx, params in enumerate(search_space, start=1):
        fold_scores = []
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
            candidate_config = dataclass_replace(config, **params)
            model = build_model(len(selected), candidate_config, {k: len(v.classes) for k, v in train_confounders.items()}).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=candidate_config.learning_rate, weight_decay=candidate_config.weight_decay)
            train_loader, train_names = make_dataloader(
                X_train[inner_train_idx], y_train[inner_train_idx],
                {k: v.values[inner_train_idx] for k, v in train_confounders.items()},
                {k: v.values[inner_train_idx] for k, v in train_envs.items()},
                candidate_config.batch_size, True,
            )
            valid_loader, valid_names = make_dataloader(
                X_train[inner_valid_idx], y_train[inner_valid_idx],
                {k: v.values[inner_valid_idx] for k, v in train_confounders.items()},
                {k: v.values[inner_valid_idx] for k, v in train_envs.items()},
                candidate_config.batch_size, False,
            )
            best_inner_auc, patience = -np.inf, 0
            for _ in range(min(80, candidate_config.max_epochs)):
                train_one_epoch(model, train_loader, train_names, opt, device, candidate_config)
                metrics, *_ = evaluate_model(model, valid_loader, valid_names, device, candidate_config)
                score = metrics["auroc"] if not np.isnan(metrics["auroc"]) else 0.5
                if score > best_inner_auc:
                    best_inner_auc, patience = score, 0
                else:
                    patience += 1
                if patience >= 12:
                    break
            fold_scores.append(best_inner_auc)
            selection_log.append({"seed": seed, "outer_fold": outer_fold, "candidate_index": candidate_idx, "inner_fold": inner_fold, "mean_inner_auc": float(np.mean(fold_scores)), "params_json": json.dumps(params, sort_keys=True)})
        score = float(np.mean(fold_scores))
        if best is None or score > best[0]:
            best = (score, params)
    assert best is not None
    chosen_params = best[1]
    final_config = dataclass_replace(config, **chosen_params)
    model = build_model(len(selected), final_config, {k: len(v.classes) for k, v in train_confounders.items()}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=final_config.learning_rate, weight_decay=final_config.weight_decay)

    train_loader, train_names = make_dataloader(X_train, y_train, {k: v.values for k, v in train_confounders.items()}, {k: v.values for k, v in train_envs.items()}, final_config.batch_size, True)
    test_loader, test_names = make_dataloader(X_test, y_test, test_confounders, test_envs, final_config.batch_size, False)

    history = []
    best_state, best_valid, patience = None, -np.inf, 0
    holdout_train_idx, holdout_valid_idx = inner_splits[0]
    valid_loader, valid_names = make_dataloader(X_train[holdout_valid_idx], y_train[holdout_valid_idx], {k: v.values[holdout_valid_idx] for k, v in train_confounders.items()}, {k: v.values[holdout_valid_idx] for k, v in train_envs.items()}, final_config.batch_size, False)
    full_train_loader, full_train_names = make_dataloader(X_train[holdout_train_idx], y_train[holdout_train_idx], {k: v.values[holdout_train_idx] for k, v in train_confounders.items()}, {k: v.values[holdout_train_idx] for k, v in train_envs.items()}, final_config.batch_size, True)
    for epoch in range(1, final_config.max_epochs + 1):
        train_metrics = train_one_epoch(model, full_train_loader, full_train_names, optimizer, device, final_config)
        valid_metrics, *_ = evaluate_model(model, valid_loader, valid_names, device, final_config)
        epoch_row = {"seed": seed, "outer_fold": outer_fold, "epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(epoch_row)
        score = valid_metrics.get("auroc", np.nan)
        score = 0.5 if np.isnan(score) else score
        if score > best_valid:
            best_valid, patience = score, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= final_config.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, y_true, y_prob, test_latent = evaluate_model(model, test_loader, test_names, device, final_config)
    test_pred = (y_prob >= 0.5).astype(int)
    oof = cohort_test[[config.sample_column, config.patient_column, config.label_column]].copy()
    oof["seed"] = seed
    oof["outer_fold"] = outer_fold
    oof["y_true"] = y_true
    oof["y_prob"] = y_prob
    oof["y_pred"] = test_pred

    gate_values = model.gate.gate_values().detach().cpu().numpy()
    gate_df = pd.DataFrame({"gene": selected, "gate_weight": gate_values, "abs_gate_weight": np.abs(gate_values), "selected_flag": (np.abs(gate_values) >= np.quantile(np.abs(gate_values), 0.9)).astype(int), "outer_fold": outer_fold, "seed": seed})

    latent_df = cohort_test[[config.sample_column, config.patient_column, config.label_column]].copy()
    latent_df["seed"] = seed
    latent_df["outer_fold"] = outer_fold
    for i in range(test_latent.shape[1]):
        latent_df[f"latent_{i + 1}"] = test_latent[:, i]
    for column in config.environment_columns:
        if column in cohort_test.columns:
            latent_df[column] = cohort_test[column].values

    metrics_row = {"seed": seed, "outer_fold": outer_fold, "n_train": len(train_idx), "n_test": len(test_idx), "selected_genes": len(selected), "gate_sparsity_fraction": float((np.abs(gate_values) < 0.1).mean()), **test_metrics}
    invariance_rows = []
    env_perf = compute_environment_metrics(oof.merge(cohort_test[[config.sample_column] + [c for c in config.environment_columns if c in cohort_test.columns]], on=config.sample_column, how="left"), config.environment_columns, seed, outer_fold)
    for item in env_skips + conf_skips:
        invariance_rows.append({"seed": seed, "outer_fold": outer_fold, "environment_or_confounder": item["column"], "status": "skipped", "reason": item["reason"]})
    for column, subset in env_perf.groupby("environment_column"):
        if subset["auroc"].notna().sum() >= 2:
            invariance_rows.append({"seed": seed, "outer_fold": outer_fold, "environment_or_confounder": column, "status": "used", "reason": "evaluated", "environment_risk_variance": float(np.nanvar(subset["balanced_accuracy"])), "environment_auroc_spread": float(np.nanmax(subset["auroc"]) - np.nanmin(subset["auroc"]))})

    if config.save_fold_models:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "genes": list(selected), "config": asdict(final_config)}, model_dir / f"deep_model_seed{seed}_fold{outer_fold}.pt")

    return pd.DataFrame([metrics_row]), oof, pd.DataFrame(history), pd.DataFrame(selection_log).assign(selected_best=lambda d: d["params_json"].eq(json.dumps(chosen_params, sort_keys=True))), gate_df, latent_df, env_perf, pd.DataFrame(invariance_rows)


def dataclass_replace(config: RunConfig, **updates: Any) -> RunConfig:
    data = asdict(config)
    data.update(updates)
    return RunConfig(**data)


def compute_environment_metrics(pred_df: pd.DataFrame, environment_columns: Sequence[str], seed: int, outer_fold: int, min_size: int = 5) -> pd.DataFrame:
    rows = []
    for column in environment_columns:
        if column not in pred_df.columns:
            rows.append({"seed": seed, "outer_fold": outer_fold, "environment_column": column, "environment_level": "missing", "n_samples": 0, "status": "missing"})
            continue
        working = pred_df[[column, "y_true", "y_prob"]].copy()
        working[column] = working[column].fillna("Unknown").astype(str)
        for level, subset in working.groupby(column):
            if len(subset) < min_size or subset["y_true"].nunique() < 2:
                rows.append({"seed": seed, "outer_fold": outer_fold, "environment_column": column, "environment_level": level, "n_samples": len(subset), "status": "skipped_small"})
                continue
            metrics = compute_binary_metrics(subset["y_true"].to_numpy(), subset["y_prob"].to_numpy())
            rows.append({"seed": seed, "outer_fold": outer_fold, "environment_column": column, "environment_level": level, "n_samples": len(subset), "status": "evaluated", **metrics})
    return pd.DataFrame(rows)


def summarise_metrics(fold_metrics: pd.DataFrame, oof_df: pd.DataFrame) -> pd.DataFrame:
    pooled = compute_binary_metrics(oof_df["y_true"].to_numpy(), oof_df["y_prob"].to_numpy())
    rows = []
    for metric in ["auroc", "auprc", "balanced_accuracy", "mcc", "f1", "sensitivity", "specificity", "brier_score"]:
        rows.append({"model_name": "SparseInvariantAAE", "summary_type": "fold_seed_mean", "metric": metric, "value": float(fold_metrics[metric].mean()), "std": float(fold_metrics[metric].std(ddof=0))})
    for metric, value in pooled.items():
        if metric in {"tn", "fp", "fn", "tp"}:
            continue
        rows.append({"model_name": "SparseInvariantAAE", "summary_type": "pooled_oof", "metric": metric, "value": float(value), "std": np.nan})
    return pd.DataFrame(rows)


def compare_with_baselines(baseline_summary: pd.DataFrame, deep_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_summary.copy()
    if {"metric", "value"}.issubset(baseline.columns):
        baseline_pivot = baseline.pivot_table(index="model_name", columns="metric", values="value", aggfunc="first").reset_index()
    else:
        baseline_pivot = baseline
    deep_pivot = deep_summary[deep_summary["summary_type"] == "pooled_oof"].pivot_table(index="model_name", columns="metric", values="value", aggfunc="first").reset_index()
    best_baseline = baseline_pivot.sort_values(["auroc", "auprc", "balanced_accuracy", "mcc"], ascending=False).head(1).copy()
    comparison = pd.concat([best_baseline.assign(model_family="baseline"), deep_pivot.assign(model_family="deep")], ignore_index=True, sort=False)
    if "auroc" in comparison.columns:
        best_val = float(best_baseline["auroc"].iloc[0]) if not best_baseline.empty else np.nan
        comparison["delta_vs_best_baseline_auroc"] = comparison["auroc"] - best_val
    return comparison


def _best_baseline_curve(baseline_summary, baseline_oof):
    if {"metric", "value"}.issubset(baseline_summary.columns):
        pivot = baseline_summary.pivot_table(index="model_name", columns="metric", values="value", aggfunc="first").reset_index()
    else:
        pivot = baseline_summary.copy()
    best_model = pivot.sort_values(["auroc", "auprc"], ascending=False)["model_name"].iloc[0]
    return best_model, baseline_oof[baseline_oof["model_name"] == best_model].copy()


def plot_deep_vs_baselines(baseline_summary, deep_summary, figures_dir: Path):
    palette = get_nature_palette()
    if {"metric", "value"}.issubset(baseline_summary.columns):
        base = baseline_summary.pivot_table(index="model_name", columns="metric", values="value", aggfunc="first").reset_index()
    else:
        base = baseline_summary.copy()
    deep = deep_summary[deep_summary["summary_type"] == "pooled_oof"].pivot_table(index="model_name", columns="metric", values="value", aggfunc="first").reset_index()
    plot_df = pd.concat([base, deep], ignore_index=True, sort=False)
    metrics = ["auroc", "auprc", "balanced_accuracy", "mcc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        ordered = plot_df.sort_values(metric, ascending=False)
        colors = [palette["tumor"] if m == "SparseInvariantAAE" else palette["normal"] for m in ordered["model_name"]]
        ax.barh(ordered["model_name"], ordered[metric], color=colors)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("Score")
        apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure5_deep_vs_baselines.svg")
    plt.close(fig)


def plot_curve_comparison(oof_df, baseline_oof, figures_dir: Path, curve_type: str):
    palette = get_nature_palette()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if curve_type == "roc":
        x1, y1, _ = roc_curve(oof_df["y_true"], oof_df["y_prob"])
        x2, y2, _ = roc_curve(baseline_oof["y_true"], baseline_oof["y_prob"])
        ax.plot(x1, y1, color=palette["tumor"], label="SparseInvariantAAE")
        ax.plot(x2, y2, color=palette["normal"], label="Best baseline")
        ax.plot([0, 1], [0, 1], linestyle="--", color=palette["unknown"])
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        title = "figure5_deep_roc.svg"
    else:
        x1, y1, _ = precision_recall_curve(oof_df["y_true"], oof_df["y_prob"])
        x2, y2, _ = precision_recall_curve(baseline_oof["y_true"], baseline_oof["y_prob"])
        ax.plot(y1, x1, color=palette["tumor"], label="SparseInvariantAAE")
        ax.plot(y2, x2, color=palette["normal"], label="Best baseline")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title = "figure5_deep_pr.svg"
    ax.legend()
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / title)
    plt.close(fig)


def plot_calibration(oof_df, baseline_oof, figures_dir: Path):
    palette = get_nature_palette()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for label, df, color in [("SparseInvariantAAE", oof_df, palette["tumor"]), ("Best baseline", baseline_oof, palette["normal"] )]:
        frac_pos, mean_pred = calibration_curve(df["y_true"], df["y_prob"], n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=color, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color=palette["unknown"])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed tumor fraction")
    ax.legend()
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure5_calibration.svg")
    plt.close(fig)


def _compute_embedding(latent_df: pd.DataFrame, latent_columns: list[str], run_umap: bool):
    X = latent_df[latent_columns].to_numpy()
    method = "PCA"
    if run_umap and umap is not None and len(latent_df) >= 10:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        method = "UMAP"
    else:
        emb = PCA(n_components=2, random_state=42).fit_transform(X)
    return emb, method


def plot_latent_space(latent_df: pd.DataFrame, color_column: str, output_name: str, figures_dir: Path, run_umap: bool):
    if color_column not in latent_df.columns:
        return
    latent_columns = [c for c in latent_df.columns if c.startswith("latent_")]
    if len(latent_columns) < 2:
        return
    palette = get_nature_palette()
    emb, method = _compute_embedding(latent_df, latent_columns, run_umap)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    working = latent_df.copy()
    working[["emb1", "emb2"]] = emb
    working[color_column] = working[color_column].fillna("Unknown").astype(str)
    for level, subset in working.groupby(color_column):
        key = str(level).lower()
        if "tumor" in key or key == "1":
            color = palette["tumor"]
        elif "normal" in key or key == "0":
            color = palette["normal"]
        elif "escc" in key:
            color = palette["escc"]
        elif "eac" in key:
            color = palette["eac"]
        elif "smok" in key:
            color = palette["smoker"]
        else:
            color = palette["unknown"]
        ax.scatter(subset["emb1"], subset["emb2"], s=28, alpha=0.85, color=color, label=level)
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(loc="best", ncol=1)
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / output_name)
    plt.close(fig)


def plot_gate_distribution(gate_df: pd.DataFrame, figures_dir: Path):
    palette = get_nature_palette()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.hist(gate_df["gate_weight"], bins=30, color=palette["tumor"], alpha=0.9)
    ax.set_xlabel("Gate weight")
    ax.set_ylabel("Genes")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure5_gate_weight_distribution.svg")
    plt.close(fig)


def plot_top_gate_genes(gate_df: pd.DataFrame, figures_dir: Path):
    palette = get_nature_palette()
    top = gate_df.groupby("gene", as_index=False)["abs_gate_weight"].mean().sort_values("abs_gate_weight", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.barh(top["gene"][::-1], top["abs_gate_weight"][::-1], color=palette["tumor"])
    ax.set_xlabel("Mean |gate weight|")
    ax.set_ylabel("Gene")
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure5_top_gate_genes.svg")
    plt.close(fig)


def plot_environment_robustness(env_df: pd.DataFrame, figures_dir: Path):
    usable = env_df[env_df.get("status", "") == "evaluated"].copy()
    if usable.empty:
        return
    palette = get_nature_palette()
    summary = usable.groupby("environment_column", as_index=False)["balanced_accuracy"].mean()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.bar(summary["environment_column"], summary["balanced_accuracy"], color=palette["eac"])
    ax.set_xlabel("Environment")
    ax.set_ylabel("Balanced accuracy")
    ax.tick_params(axis="x", rotation=30)
    apply_text_style(ax)
    save_publication_figure(fig, figures_dir / "figure5_environment_robustness.svg")
    plt.close(fig)


def plot_training_dynamics(history_df: pd.DataFrame, figures_dir: Path):
    if history_df.empty:
        return
    palette = get_nature_palette()
    rep = history_df.sort_values(["seed", "outer_fold", "epoch"]).groupby(["seed", "outer_fold"], as_index=False).head(1).iloc[0]
    subset = history_df[(history_df["seed"] == rep["seed"]) & (history_df["outer_fold"] == rep["outer_fold"])]
    fig, ax1 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax1.plot(subset["epoch"], subset["train_total_loss"], color=palette["train"], label="Train loss")
    ax1.plot(subset["epoch"], subset["valid_total_loss"], color=palette["validation"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(subset["epoch"], subset["valid_auroc"], color=palette["tumor"], label="Validation AUROC")
    ax2.set_ylabel("AUROC")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    apply_text_style(ax1)
    apply_text_style(ax2)
    save_publication_figure(fig, figures_dir / "figure5_training_dynamics.svg")
    plt.close(fig)


def save_outputs(output_dir: Path, fold_metrics, summary_metrics, oof_df, history_df, selection_log, gate_df, latent_df, env_perf, invariance_df, baseline_vs_deep, config: RunConfig):
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(output_dir / "deep_fold_metrics.csv", index=False)
    summary_metrics.to_csv(output_dir / "deep_summary_metrics.csv", index=False)
    oof_df.to_csv(output_dir / "deep_oof_predictions.csv", index=False)
    history_df.to_csv(output_dir / "deep_training_history.csv", index=False)
    selection_log.to_csv(output_dir / "deep_model_selection_log.csv", index=False)
    gate_df.to_csv(output_dir / "gate_weights.csv", index=False)
    if not latent_df.empty:
        latent_df.to_csv(output_dir / "latent_embeddings.csv", index=False)
    env_perf.to_csv(output_dir / "environment_performance.csv", index=False)
    invariance_df.to_csv(output_dir / "invariance_summary.csv", index=False)
    baseline_vs_deep.to_csv(output_dir / "baseline_vs_deep_comparison.csv", index=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)


def update_readme_step4():
    return None


def main() -> None:
    args = parse_args()
    config = build_run_config(args)
    configure_publication_plotting()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cohort_df, normalized_df, folds_df, baseline_summary, baseline_oof = load_inputs(config)
    cohort_df, normalized_df = align_inputs(cohort_df, normalized_df, folds_df, config)
    outer_folds = validate_grouped_folds(cohort_df, config)

    all_fold_metrics, all_oof, all_history, all_selection, all_gates, all_latent, all_env, all_invariance = ([] for _ in range(8))
    for seed in config.random_seeds:
        set_random_seed(seed)
        for outer_fold in outer_folds:
            outputs = fit_one_outer_fold(cohort_df, normalized_df, config, outer_fold, seed, output_dir, device)
            for collection, item in zip([all_fold_metrics, all_oof, all_history, all_selection, all_gates, all_latent, all_env, all_invariance], outputs):
                collection.append(item)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)
    history_df = pd.concat(all_history, ignore_index=True)
    selection_log = pd.concat(all_selection, ignore_index=True)
    gate_df = pd.concat(all_gates, ignore_index=True)
    latent_df = pd.concat(all_latent, ignore_index=True) if config.save_latent_embeddings and all_latent else pd.DataFrame()
    env_perf = pd.concat(all_env, ignore_index=True)
    invariance_df = pd.concat(all_invariance, ignore_index=True)
    summary_metrics = summarise_metrics(fold_metrics, oof_df)
    baseline_vs_deep = compare_with_baselines(baseline_summary, summary_metrics)
    save_outputs(output_dir, fold_metrics, summary_metrics, oof_df, history_df, selection_log, gate_df, latent_df, env_perf, invariance_df, baseline_vs_deep, config)

    plot_deep_vs_baselines(baseline_summary, summary_metrics, figures_dir)
    best_baseline_name, best_baseline_oof = _best_baseline_curve(baseline_summary, baseline_oof)
    plot_curve_comparison(oof_df, best_baseline_oof, figures_dir, "roc")
    plot_curve_comparison(oof_df, best_baseline_oof, figures_dir, "pr")
    plot_calibration(oof_df, best_baseline_oof, figures_dir)
    if not latent_df.empty:
        label_map = {0: "normal", 1: "tumor"}
        latent_for_plot = latent_df.copy()
        latent_for_plot[config.label_column] = latent_for_plot[config.label_column].map(label_map).fillna(latent_for_plot[config.label_column])
        plot_latent_space(latent_for_plot, config.label_column, "figure5_latent_space_tumor_normal.svg", figures_dir, config.run_umap)
        plot_latent_space(latent_for_plot, "env_histology", "figure5_latent_space_histology.svg", figures_dir, config.run_umap)
        plot_latent_space(latent_for_plot, "env_smoking", "figure5_latent_space_smoking.svg", figures_dir, config.run_umap)
    plot_gate_distribution(gate_df, figures_dir)
    plot_top_gate_genes(gate_df, figures_dir)
    plot_environment_robustness(env_perf, figures_dir)
    plot_training_dynamics(history_df, figures_dir)

    deep_pooled = summary_metrics[summary_metrics["summary_type"] == "pooled_oof"].set_index("metric")["value"].to_dict()
    baseline_best = compare_with_baselines(baseline_summary, summary_metrics).query("model_family == 'baseline'")
    baseline_auroc = float(baseline_best["auroc"].iloc[0]) if not baseline_best.empty and "auroc" in baseline_best else np.nan
    print(
        f"Deep AUROC={deep_pooled.get('auroc', np.nan):.4f} | "
        f"AUPRC={deep_pooled.get('auprc', np.nan):.4f} | "
        f"BalancedAcc={deep_pooled.get('balanced_accuracy', np.nan):.4f} | "
        f"MCC={deep_pooled.get('mcc', np.nan):.4f} | "
        f"BestBaselineAUROC={baseline_auroc:.4f} | Delta={deep_pooled.get('auroc', np.nan) - baseline_auroc:.4f} | "
        f"AvgSelectedGenes={fold_metrics['selected_genes'].mean():.1f} | Folds={len(outer_folds)} | Seeds={len(config.random_seeds)} | Samples={len(oof_df)}"
    )


if __name__ == "__main__":
    main()
