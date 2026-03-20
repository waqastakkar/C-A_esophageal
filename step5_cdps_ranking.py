#!/usr/bin/env python3
"""Step 5: Candidate Driver Priority Score (CDPS) ranking for TCGA ESCA.

This module implements the first major Stage III analysis block:
candidate driver prioritization and interpretability for the Step 4 deep
invariant sparse model. The implementation combines attribution, gate evidence,
selection stability, environment consistency, and in silico perturbation impact
into a reproducible Candidate Driver Priority Score (CDPS).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from deep_model_utils import SparseInvariantAdversarialAutoencoderClassifier, choose_hidden_dims, set_random_seed
from publication_style import apply_text_style, configure_publication_plotting, get_nature_palette, save_publication_figure


@dataclass(frozen=True)
class Step5Config:
    cohort: str
    normalized: str
    folds: str
    deep_oof: str
    gate_weights: str
    latent_embeddings: str | None
    deep_history: str | None
    deep_metrics: str
    deep_model_dir: str | None
    output_dir: str
    label_column: str
    patient_column: str
    sample_column: str
    outer_fold_column: str
    environment_columns: list[str]
    random_seeds: list[int]
    bootstrap_repeats: int
    top_genes_for_perturbation: int
    top_genes_for_detailed_report: int
    attribution_methods: list[str]
    perturbation_quantile: float
    min_subgroup_size: int
    save_samplewise_attributions: bool
    run_pathway_summary: bool
    gene_annotation: str | None
    pathway_gmt: str | None
    use_absolute_attribution: bool
    normalize_component_scores: bool
    weight_attribution: float
    weight_gate: float
    weight_stability: float
    weight_invariance: float
    weight_perturbation: float
    attribution_sample_cap_per_class: int
    integrated_gradients_steps: int
    perturbation_reference: str
    attribution_batch_size: int
    device: str


@dataclass
class LoadedModelArtifact:
    seed: int
    outer_fold: int
    genes: list[str]
    model: SparseInvariantAdversarialAutoencoderClassifier
    config_dict: dict[str, Any]
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--normalized", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--deep-oof", required=True)
    parser.add_argument("--gate-weights", required=True)
    parser.add_argument("--latent-embeddings")
    parser.add_argument("--deep-history")
    parser.add_argument("--deep-metrics", required=True)
    parser.add_argument("--deep-model-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="disease_label")
    parser.add_argument("--patient-column", default="patient_id")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--outer-fold-column", default="outer_fold")
    parser.add_argument("--environment-columns", default="env_sex,env_smoking,env_histology,env_stage,env_country_or_region")
    parser.add_argument("--random-seeds", default="42,52,62,72,82")
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--top-genes-for-perturbation", type=int, default=100)
    parser.add_argument("--top-genes-for-detailed-report", type=int, default=25)
    parser.add_argument("--attribution-methods", default="integrated_gradients,gradient_x_input,gate_weight")
    parser.add_argument("--perturbation-quantile", type=float, default=0.1)
    parser.add_argument("--min-subgroup-size", type=int, default=10)
    parser.add_argument("--save-samplewise-attributions", action="store_true")
    parser.add_argument("--run-pathway-summary", action="store_true")
    parser.add_argument("--gene-annotation")
    parser.add_argument("--pathway-gmt")
    parser.add_argument("--use-absolute-attribution", action="store_true")
    parser.add_argument("--normalize-component-scores", action="store_true")
    parser.add_argument("--weight-attribution", type=float, default=0.30)
    parser.add_argument("--weight-gate", type=float, default=0.20)
    parser.add_argument("--weight-stability", type=float, default=0.20)
    parser.add_argument("--weight-invariance", type=float, default=0.15)
    parser.add_argument("--weight-perturbation", type=float, default=0.15)
    parser.add_argument("--attribution-sample-cap-per-class", type=int, default=25)
    parser.add_argument("--integrated-gradients-steps", type=int, default=32)
    parser.add_argument("--perturbation-reference", choices=["quantile", "median"], default="quantile")
    parser.add_argument("--attribution-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Step5Config:
    return Step5Config(
        cohort=args.cohort,
        normalized=args.normalized,
        folds=args.folds,
        deep_oof=args.deep_oof,
        gate_weights=args.gate_weights,
        latent_embeddings=args.latent_embeddings,
        deep_history=args.deep_history,
        deep_metrics=args.deep_metrics,
        deep_model_dir=args.deep_model_dir,
        output_dir=args.output_dir,
        label_column=args.label_column,
        patient_column=args.patient_column,
        sample_column=args.sample_column,
        outer_fold_column=args.outer_fold_column,
        environment_columns=[c.strip() for c in args.environment_columns.split(",") if c.strip()],
        random_seeds=[int(x.strip()) for x in args.random_seeds.split(",") if x.strip()],
        bootstrap_repeats=int(args.bootstrap_repeats),
        top_genes_for_perturbation=int(args.top_genes_for_perturbation),
        top_genes_for_detailed_report=int(args.top_genes_for_detailed_report),
        attribution_methods=[m.strip() for m in args.attribution_methods.split(",") if m.strip()],
        perturbation_quantile=float(args.perturbation_quantile),
        min_subgroup_size=int(args.min_subgroup_size),
        save_samplewise_attributions=bool(args.save_samplewise_attributions),
        run_pathway_summary=bool(args.run_pathway_summary),
        gene_annotation=args.gene_annotation,
        pathway_gmt=args.pathway_gmt,
        use_absolute_attribution=bool(args.use_absolute_attribution),
        normalize_component_scores=bool(args.normalize_component_scores),
        weight_attribution=float(args.weight_attribution),
        weight_gate=float(args.weight_gate),
        weight_stability=float(args.weight_stability),
        weight_invariance=float(args.weight_invariance),
        weight_perturbation=float(args.weight_perturbation),
        attribution_sample_cap_per_class=int(args.attribution_sample_cap_per_class),
        integrated_gradients_steps=int(args.integrated_gradients_steps),
        perturbation_reference=str(args.perturbation_reference),
        attribution_batch_size=int(args.attribution_batch_size),
        device=str(args.device),
    )


def validate_required_columns(df: pd.DataFrame, required: Sequence[str], frame_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{frame_name} missing required columns: {missing}")


def load_inputs(config: Step5Config) -> dict[str, pd.DataFrame]:
    cohort = pd.read_csv(config.cohort)
    normalized = pd.read_csv(config.normalized)
    folds = pd.read_csv(config.folds)
    deep_oof = pd.read_csv(config.deep_oof)
    gate_weights = pd.read_csv(config.gate_weights)
    deep_metrics = pd.read_csv(config.deep_metrics)
    latent = pd.read_csv(config.latent_embeddings) if config.latent_embeddings and Path(config.latent_embeddings).exists() else pd.DataFrame()
    history = pd.read_csv(config.deep_history) if config.deep_history and Path(config.deep_history).exists() else pd.DataFrame()
    gene_annotation = pd.read_csv(config.gene_annotation) if config.gene_annotation and Path(config.gene_annotation).exists() else pd.DataFrame()

    validate_required_columns(cohort, [config.sample_column, config.patient_column, config.label_column], "cohort")
    validate_required_columns(normalized, [config.sample_column], "normalized")
    validate_required_columns(folds, [config.sample_column, config.patient_column, config.outer_fold_column], "folds")
    validate_required_columns(deep_oof, [config.sample_column, "seed", "outer_fold", "y_prob"], "deep_oof")
    validate_required_columns(gate_weights, ["gene", "gate_weight", "outer_fold", "seed"], "gate_weights")

    return {
        "cohort": cohort,
        "normalized": normalized,
        "folds": folds,
        "deep_oof": deep_oof,
        "gate_weights": gate_weights,
        "deep_metrics": deep_metrics,
        "latent": latent,
        "history": history,
        "gene_annotation": gene_annotation,
    }


def align_inputs(inputs: dict[str, pd.DataFrame], config: Step5Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = inputs["cohort"].copy()
    normalized = inputs["normalized"].copy()
    folds = inputs["folds"].copy()
    for frame in (cohort, normalized, folds):
        frame[config.sample_column] = frame[config.sample_column].astype(str)
    shared_ids = sorted(set(cohort[config.sample_column]) & set(normalized[config.sample_column]) & set(folds[config.sample_column]))
    if not shared_ids:
        raise ValueError("No shared sample identifiers across cohort, normalized matrix, and fold table.")
    cohort = cohort[cohort[config.sample_column].isin(shared_ids)].copy()
    normalized = normalized[normalized[config.sample_column].isin(shared_ids)].copy()
    folds = folds[folds[config.sample_column].isin(shared_ids)].copy()
    merged = cohort.merge(
        folds.drop(columns=[config.label_column], errors="ignore"),
        on=[config.sample_column, config.patient_column],
        how="inner",
    )
    merged = merged.sort_values([config.outer_fold_column, config.sample_column], kind="mergesort").reset_index(drop=True)
    normalized = normalized.set_index(config.sample_column).loc[merged[config.sample_column]].reset_index()
    return merged, normalized, inputs["deep_oof"].copy()


def infer_adversary_specs(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    specs: dict[str, int] = {}
    pattern = re.compile(r"^adversaries\.(?P<name>.+)\.3\.bias$")
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            specs[match.group("name")] = int(value.shape[0])
    return specs


def load_step4_models(config: Step5Config) -> list[LoadedModelArtifact]:
    if not config.deep_model_dir:
        print("[Step5] No --deep-model-dir provided. Attribution and perturbation will fall back to gate-only summaries where needed.")
        return []
    model_dir = Path(config.deep_model_dir)
    if not model_dir.exists():
        print(f"[Step5] Deep model directory not found: {model_dir}. Proceeding without checkpoints.")
        return []

    artifacts: list[LoadedModelArtifact] = []
    for path in sorted(model_dir.glob("deep_model_seed*_fold*.pt")):
        match = re.search(r"seed(?P<seed>\d+)_fold(?P<fold>\d+)", path.stem)
        if not match:
            continue
        payload = torch.load(path, map_location="cpu")
        state_dict = payload["state_dict"]
        genes = [str(gene) for gene in payload.get("genes", [])]
        config_dict = payload.get("config", {})
        latent_dim = int(config_dict.get("latent_dim", 32))
        use_l1_gate = bool(config_dict.get("use_l1_gate", True))
        use_hard_concrete_gate = bool(config_dict.get("use_hard_concrete_gate", False))
        adversary_specs = infer_adversary_specs(state_dict)
        model = SparseInvariantAdversarialAutoencoderClassifier(
            input_dim=len(genes),
            latent_dim=latent_dim,
            hidden_dims=choose_hidden_dims(len(genes), latent_dim),
            dropout=0.2,
            adversary_specs=adversary_specs,
            grl_lambda=float(config_dict.get("adversary_weight", 1.0)),
            use_l1_gate=use_l1_gate,
            use_hard_concrete_gate=use_hard_concrete_gate,
        )
        model.load_state_dict(state_dict)
        model.eval()
        artifacts.append(
            LoadedModelArtifact(
                seed=int(match.group("seed")),
                outer_fold=int(match.group("fold")),
                genes=genes,
                model=model,
                config_dict=config_dict,
                path=path,
            )
        )
    print(f"[Step5] Loaded {len(artifacts)} Step 4 checkpoint(s) from {model_dir}.")
    return artifacts


def aggregate_gate_weights(gate_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = (
        gate_weights.groupby(["seed", "outer_fold", "gene"], as_index=False)
        .agg(
            gate_weight=("gate_weight", "mean"),
            abs_gate_weight=("abs_gate_weight", "mean"),
            selected_flag=("selected_flag", "max"),
        )
    )
    summary = (
        per_run.groupby("gene", as_index=False)
        .agg(
            mean_gate_weight=("gate_weight", "mean"),
            mean_abs_gate_weight=("abs_gate_weight", "mean"),
            gate_selection_frequency=("selected_flag", "mean"),
            fold_count=("outer_fold", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values("mean_abs_gate_weight", ascending=False)
    )
    return per_run, summary


def _balanced_sample_subset(frame: pd.DataFrame, label_column: str, cap_per_class: int, seed: int) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(seed)
    for _, subset in frame.groupby(label_column):
        if len(subset) <= cap_per_class:
            sampled_parts.append(subset.copy())
            continue
        idx = rng.choice(subset.index.to_numpy(), size=cap_per_class, replace=False)
        sampled_parts.append(subset.loc[idx].copy())
    if not sampled_parts:
        return frame.iloc[0:0].copy()
    return pd.concat(sampled_parts, ignore_index=True)


@torch.no_grad()
def _predict_probability_and_latent(model: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(batch)
    probs = torch.sigmoid(outputs["logits"])
    latent = outputs["z"]
    return probs, latent


def compute_integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    if inputs.numel() == 0:
        return inputs.detach().clone()
    scaled_diffs = inputs - baseline
    total_gradients = torch.zeros_like(inputs)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=inputs.device)[1:]:
        current = (baseline + alpha * scaled_diffs).detach().clone().requires_grad_(True)
        logits = model(current)["logits"].sum()
        gradients = torch.autograd.grad(logits, current, retain_graph=False, create_graph=False)[0]
        total_gradients += gradients.detach()
    return scaled_diffs * total_gradients / steps


def compute_gradient_x_input(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    current = inputs.detach().clone().requires_grad_(True)
    logits = model(current)["logits"].sum()
    gradients = torch.autograd.grad(logits, current, retain_graph=False, create_graph=False)[0]
    return current.detach() * gradients.detach()


def compute_attributions_for_artifact(
    artifact: LoadedModelArtifact,
    merged: pd.DataFrame,
    normalized: pd.DataFrame,
    config: Step5Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_mask = merged[config.outer_fold_column].astype(int).eq(artifact.outer_fold)
    fold_samples = merged.loc[sample_mask, [config.sample_column, config.label_column, config.outer_fold_column]].copy()
    fold_samples = fold_samples[fold_samples[config.sample_column].isin(normalized[config.sample_column])]
    fold_samples = _balanced_sample_subset(fold_samples, config.label_column, config.attribution_sample_cap_per_class, artifact.seed)
    if fold_samples.empty:
        return pd.DataFrame(), pd.DataFrame()

    available_genes = [gene for gene in artifact.genes if gene in normalized.columns]
    if not available_genes:
        print(f"[Step5] No overlapping genes found for checkpoint {artifact.path.name}; skipping attribution.")
        return pd.DataFrame(), pd.DataFrame()

    x_frame = normalized.set_index(config.sample_column).loc[fold_samples[config.sample_column], available_genes]
    x_np = x_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    x_tensor = torch.tensor(x_np, dtype=torch.float32, device=config.device)
    baseline_np = np.zeros_like(x_np)
    baseline_tensor = torch.tensor(baseline_np, dtype=torch.float32, device=config.device)

    method_rows: list[pd.DataFrame] = []
    samplewise_rows: list[pd.DataFrame] = []
    artifact.model.to(config.device)
    for method in config.attribution_methods:
        if method == "integrated_gradients":
            values = compute_integrated_gradients(artifact.model, x_tensor, baseline_tensor, config.integrated_gradients_steps).detach().cpu().numpy()
        elif method == "gradient_x_input":
            values = compute_gradient_x_input(artifact.model, x_tensor).detach().cpu().numpy()
        elif method == "gate_weight":
            gate = artifact.model.gate.gate_values().detach().cpu().numpy()
            values = np.tile(gate[None, :], (x_np.shape[0], 1))
        else:
            print(f"[Step5] Unsupported attribution method '{method}' requested; skipping.")
            continue

        frame = pd.DataFrame(values, columns=available_genes)
        frame.insert(0, config.sample_column, fold_samples[config.sample_column].to_numpy())
        frame["seed"] = artifact.seed
        frame["outer_fold"] = artifact.outer_fold
        frame["attribution_method"] = method
        long_frame = frame.melt(
            id_vars=[config.sample_column, "seed", "outer_fold", "attribution_method"],
            var_name="gene",
            value_name="attribution_value",
        )
        samplewise_rows.append(long_frame)

        summary = (
            long_frame.groupby("gene", as_index=False)
            .agg(
                mean_attribution=("attribution_value", "mean"),
                mean_abs_attribution=("attribution_value", lambda s: float(np.mean(np.abs(s)))),
                std_attribution=("attribution_value", "std"),
            )
            .assign(method=method, seed=artifact.seed, outer_fold=artifact.outer_fold)
        )
        method_rows.append(summary)

    samplewise = pd.concat(samplewise_rows, ignore_index=True) if samplewise_rows else pd.DataFrame()
    summary = pd.concat(method_rows, ignore_index=True) if method_rows else pd.DataFrame()
    return summary, samplewise


def aggregate_attribution_scores(per_run_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per_run_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    aggregated = (
        per_run_summary.groupby(["gene", "method"], as_index=False)
        .agg(
            mean_attribution=("mean_attribution", "mean"),
            mean_abs_attribution=("mean_abs_attribution", "mean"),
            std_attribution=("mean_attribution", "std"),
            fold_count=("outer_fold", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["method", "mean_abs_attribution"], ascending=[True, False])
    )
    by_gene = (
        aggregated.groupby("gene", as_index=False)
        .agg(
            attribution_score=("mean_abs_attribution", "mean"),
            signed_attribution_score=("mean_attribution", "mean"),
            attribution_method_count=("method", "nunique"),
        )
    )
    return aggregated, by_gene


def compute_topk_selection_frequency(run_scores: pd.DataFrame, top_k: int) -> pd.Series:
    def _freq(group: pd.DataFrame) -> float:
        if group.empty:
            return math.nan
        return float((group["rank_within_run"] <= top_k).mean())
    return run_scores.groupby("gene").apply(_freq)


def compute_stability_summary(run_scores: pd.DataFrame) -> pd.DataFrame:
    if run_scores.empty:
        return pd.DataFrame(columns=["gene", "top25_frequency", "top100_frequency", "mean_rank", "rank_std"]) 
    summary = (
        run_scores.groupby("gene", as_index=False)
        .agg(
            mean_rank=("rank_within_run", "mean"),
            rank_std=("rank_within_run", "std"),
            run_score_mean=("run_predictive_score", "mean"),
            run_score_std=("run_predictive_score", "std"),
        )
    )
    summary["top25_frequency"] = summary["gene"].map(compute_topk_selection_frequency(run_scores, 25))
    summary["top100_frequency"] = summary["gene"].map(compute_topk_selection_frequency(run_scores, 100))
    summary["importance_cv"] = summary["run_score_std"] / summary["run_score_mean"].replace(0, np.nan)
    summary["stability_score_raw"] = (
        0.4 * summary["top25_frequency"].fillna(0)
        + 0.3 * summary["top100_frequency"].fillna(0)
        + 0.2 * (1.0 / (1.0 + summary["rank_std"].fillna(summary["rank_std"].max())))
        + 0.1 * (1.0 / (1.0 + summary["importance_cv"].replace([np.inf, -np.inf], np.nan).fillna(summary["importance_cv"].median())))
    )
    return summary


def build_run_score_table(attribution_per_run: pd.DataFrame, gate_per_run: pd.DataFrame) -> pd.DataFrame:
    attr_component = pd.DataFrame()
    if not attribution_per_run.empty:
        attr_component = (
            attribution_per_run.groupby(["seed", "outer_fold", "gene"], as_index=False)
            .agg(mean_abs_attribution=("mean_abs_attribution", "mean"))
        )
    merged = gate_per_run.merge(attr_component, on=["seed", "outer_fold", "gene"], how="outer")
    merged["mean_abs_attribution"] = merged["mean_abs_attribution"].fillna(0.0)
    merged["abs_gate_weight"] = merged["abs_gate_weight"].fillna(0.0)
    merged["run_predictive_score"] = 0.6 * merged["mean_abs_attribution"] + 0.4 * merged["abs_gate_weight"]
    merged["rank_within_run"] = merged.groupby(["seed", "outer_fold"])["run_predictive_score"].rank(ascending=False, method="average")
    return merged


def compute_environment_consistency(
    samplewise_attributions: pd.DataFrame,
    merged: pd.DataFrame,
    perturbation_samplewise: pd.DataFrame,
    config: Step5Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_envs = [column for column in config.environment_columns if column in merged.columns]
    if not available_envs or samplewise_attributions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    attrib = samplewise_attributions.merge(
        merged[[config.sample_column] + available_envs],
        on=config.sample_column,
        how="left",
    )
    pert = perturbation_samplewise.copy()
    if not pert.empty:
        pert = pert.merge(merged[[config.sample_column] + available_envs], on=config.sample_column, how="left")

    effect_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    for env_column in available_envs:
        for gene, gene_frame in attrib.groupby("gene"):
            env_effects: list[float] = []
            coverage = 0
            for env_level, subset in gene_frame.groupby(env_column):
                if len(subset) < config.min_subgroup_size:
                    continue
                mean_attr = float(np.mean(np.abs(subset["attribution_value"])))
                mean_pert = np.nan
                if not pert.empty:
                    pert_subset = pert[(pert["gene"] == gene) & (pert[env_column] == env_level)]
                    if len(pert_subset) >= config.min_subgroup_size:
                        mean_pert = float(pert_subset["delta_prob"].abs().mean())
                combined_effect = np.nanmean([mean_attr, mean_pert]) if not np.isnan(mean_pert) else mean_attr
                env_effects.append(combined_effect)
                coverage += 1
                effect_rows.append(
                    {
                        "gene": gene,
                        "environment_column": env_column,
                        "environment_level": env_level,
                        "mean_attribution": mean_attr,
                        "mean_perturbation_effect": mean_pert,
                        "n_samples": int(len(subset)),
                    }
                )
            if not env_effects:
                continue
            env_effects_arr = np.asarray(env_effects, dtype=float)
            mean_effect = float(np.mean(env_effects_arr))
            std_effect = float(np.std(env_effects_arr, ddof=0))
            gap = float(np.max(env_effects_arr) - np.min(env_effects_arr)) if len(env_effects_arr) > 1 else 0.0
            invariance_score = mean_effect / (1.0 + std_effect + gap)
            summary_rows.append(
                {
                    "gene": gene,
                    "environment_column": env_column,
                    "environment_coverage": coverage,
                    "mean_environment_effect": mean_effect,
                    "std_environment_effect": std_effect,
                    "max_environment_gap": gap,
                    "invariance_score": invariance_score,
                }
            )
            heatmap_rows.append({"gene": gene, "environment_column": env_column, "invariance_score": invariance_score})
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df, pd.DataFrame(effect_rows), pd.DataFrame(heatmap_rows)
    return summary_df, pd.DataFrame(effect_rows), pd.DataFrame(heatmap_rows)


def _class_reference_values(values: pd.Series, label: pd.Series, target_label: int, quantile: float, reference: str) -> float:
    class_values = values[label.eq(target_label)].dropna()
    if class_values.empty:
        return float(values.median())
    if reference == "median":
        return float(class_values.median())
    q = quantile if target_label == 0 else 1.0 - quantile
    return float(class_values.quantile(q))


def run_gene_perturbation(
    top_genes: Sequence[str],
    artifacts: list[LoadedModelArtifact],
    merged: pd.DataFrame,
    normalized: pd.DataFrame,
    config: Step5Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not artifacts or not top_genes:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    expr = normalized.set_index(config.sample_column)
    merged_idx = merged.set_index(config.sample_column)
    samplewise_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []

    for artifact in artifacts:
        model = artifact.model.to(config.device)
        fold_ids = merged.loc[merged[config.outer_fold_column].astype(int).eq(artifact.outer_fold), config.sample_column].astype(str)
        usable_genes = [gene for gene in artifact.genes if gene in expr.columns]
        if not usable_genes:
            continue
        fold_expr = expr.loc[fold_ids, usable_genes].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        fold_meta = merged_idx.loc[fold_ids]
        x_base = torch.tensor(fold_expr.to_numpy(dtype=np.float32), dtype=torch.float32, device=config.device)
        base_prob, base_latent = _predict_probability_and_latent(model, x_base)
        base_prob_np = base_prob.detach().cpu().numpy()
        base_latent_np = base_latent.detach().cpu().numpy()
        label_series = fold_meta[config.label_column].astype(int)

        for gene in top_genes:
            if gene not in usable_genes:
                continue
            perturbed = fold_expr.copy()
            gene_values = perturbed[gene]
            normal_ref = _class_reference_values(gene_values, label_series, 0, config.perturbation_quantile, config.perturbation_reference)
            tumor_ref = _class_reference_values(gene_values, label_series, 1, config.perturbation_quantile, config.perturbation_reference)
            tumor_mask = label_series.eq(1)
            normal_mask = label_series.eq(0)
            perturbed.loc[tumor_mask, gene] = normal_ref
            perturbed.loc[normal_mask, gene] = tumor_ref

            x_pert = torch.tensor(perturbed.to_numpy(dtype=np.float32), dtype=torch.float32, device=config.device)
            pert_prob, pert_latent = _predict_probability_and_latent(model, x_pert)
            pert_prob_np = pert_prob.detach().cpu().numpy()
            pert_latent_np = pert_latent.detach().cpu().numpy()
            delta_prob = pert_prob_np - base_prob_np
            latent_shift = np.linalg.norm(pert_latent_np - base_latent_np, axis=1)
            expected_direction = np.where(label_series.to_numpy() == 1, delta_prob < 0, delta_prob > 0)

            for idx, sample_id in enumerate(fold_ids):
                samplewise_rows.append(
                    {
                        config.sample_column: sample_id,
                        "gene": gene,
                        "original_prob": float(base_prob_np[idx]),
                        "perturbed_prob": float(pert_prob_np[idx]),
                        "delta_prob": float(delta_prob[idx]),
                        "original_latent_norm": float(np.linalg.norm(base_latent_np[idx])),
                        "perturbed_latent_norm": float(np.linalg.norm(pert_latent_np[idx])),
                        "outer_fold": artifact.outer_fold,
                        "seed": artifact.seed,
                    }
                )
            summary_rows.append(
                {
                    "gene": gene,
                    "seed": artifact.seed,
                    "outer_fold": artifact.outer_fold,
                    "mean_delta_prob_tumor": float(np.mean(delta_prob[tumor_mask.to_numpy()])) if tumor_mask.any() else np.nan,
                    "mean_delta_prob_normal": float(np.mean(delta_prob[normal_mask.to_numpy()])) if normal_mask.any() else np.nan,
                    "mean_abs_delta_prob": float(np.mean(np.abs(delta_prob))),
                    "latent_shift_norm": float(np.mean(latent_shift)),
                    "directional_consistency": float(np.mean(expected_direction)),
                }
            )
            latent_rows.append(
                {
                    "gene": gene,
                    "seed": artifact.seed,
                    "outer_fold": artifact.outer_fold,
                    "latent_shift_norm": float(np.mean(latent_shift)),
                    "tumor_latent_shift_norm": float(np.mean(latent_shift[tumor_mask.to_numpy()])) if tumor_mask.any() else np.nan,
                    "normal_latent_shift_norm": float(np.mean(latent_shift[normal_mask.to_numpy()])) if normal_mask.any() else np.nan,
                }
            )

    samplewise_df = pd.DataFrame(samplewise_rows)
    per_run_summary = pd.DataFrame(summary_rows)
    latent_df = pd.DataFrame(latent_rows)
    if per_run_summary.empty:
        return per_run_summary, samplewise_df, latent_df
    aggregated = (
        per_run_summary.groupby("gene", as_index=False)
        .agg(
            mean_delta_prob_tumor=("mean_delta_prob_tumor", "mean"),
            mean_delta_prob_normal=("mean_delta_prob_normal", "mean"),
            mean_abs_delta_prob=("mean_abs_delta_prob", "mean"),
            latent_shift_norm=("latent_shift_norm", "mean"),
            directional_consistency=("directional_consistency", "mean"),
        )
    )
    aggregated["perturbation_score"] = aggregated["mean_abs_delta_prob"] * (0.5 + 0.5 * aggregated["directional_consistency"].fillna(0))
    return aggregated, samplewise_df, latent_df


def normalize_component_scores(frame: pd.DataFrame, component_columns: Sequence[str], enabled: bool) -> pd.DataFrame:
    frame = frame.copy()
    if not enabled:
        return frame
    for column in component_columns:
        values = frame[[column]].fillna(0.0)
        if values.nunique().iloc[0] <= 1:
            frame[column] = values.iloc[:, 0].to_numpy()
            continue
        scaler = MinMaxScaler()
        frame[column] = scaler.fit_transform(values).ravel()
    return frame


def compute_cdps(summary_df: pd.DataFrame, config: Step5Config) -> pd.DataFrame:
    summary_df = summary_df.copy()
    component_columns = ["attribution_score", "gate_score", "stability_score", "invariance_score", "perturbation_score"]
    summary_df[component_columns] = summary_df[component_columns].fillna(0.0)
    summary_df = normalize_component_scores(summary_df, component_columns, config.normalize_component_scores)
    summary_df["cdps_score"] = (
        config.weight_attribution * summary_df["attribution_score"]
        + config.weight_gate * summary_df["gate_score"]
        + config.weight_stability * summary_df["stability_score"]
        + config.weight_invariance * summary_df["invariance_score"]
        + config.weight_perturbation * summary_df["perturbation_score"]
    )
    return summary_df


def rank_genes(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.sort_values(["cdps_score", "attribution_score", "gate_score"], ascending=False).reset_index(drop=True)
    ranked["cdps_rank"] = np.arange(1, len(ranked) + 1)
    ranked["notes"] = np.where(
        ranked["directional_consistency"].fillna(0) >= 0.7,
        "Stable computationally prioritized driver-associated gene with consistent counterfactual effect.",
        "Computationally prioritized candidate driver gene; interpret with observational caution.",
    )
    return ranked


def bootstrap_cdps(
    run_scores: pd.DataFrame,
    gene_summary: pd.DataFrame,
    config: Step5Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if run_scores.empty or gene_summary.empty:
        return pd.DataFrame(), gene_summary
    rng = np.random.default_rng(20260319)
    run_keys = run_scores[["seed", "outer_fold"]].drop_duplicates().reset_index(drop=True)
    bootstrap_rows: list[dict[str, Any]] = []
    genes = gene_summary["gene"].tolist()

    for iteration in range(config.bootstrap_repeats):
        sampled_idx = rng.integers(0, len(run_keys), size=len(run_keys))
        sampled_runs = run_keys.iloc[sampled_idx].copy()
        sampled_runs["bootstrap_copy"] = np.arange(len(sampled_runs))
        sampled = sampled_runs.merge(run_scores, on=["seed", "outer_fold"], how="left")
        sampled_gene = sampled.groupby("gene", as_index=False)["run_predictive_score"].mean()
        sampled_gene = gene_summary[["gene", "attribution_score", "gate_score", "stability_score", "invariance_score", "perturbation_score"]].merge(sampled_gene, on="gene", how="left")
        sampled_gene["stability_score"] = sampled_gene["stability_score"].fillna(0.0) * 0.7 + sampled_gene["run_predictive_score"].fillna(0.0) * 0.3
        sampled_gene = compute_cdps(sampled_gene, config)
        sampled_gene = sampled_gene.sort_values("cdps_score", ascending=False).reset_index(drop=True)
        sampled_gene["rank_bootstrap"] = np.arange(1, len(sampled_gene) + 1)
        for _, row in sampled_gene.iterrows():
            bootstrap_rows.append(
                {
                    "gene": row["gene"],
                    "bootstrap_iteration": iteration,
                    "cdps_bootstrap": row["cdps_score"],
                    "rank_bootstrap": int(row["rank_bootstrap"]),
                }
            )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    if bootstrap_df.empty:
        return bootstrap_df, gene_summary
    stats_df = (
        bootstrap_df.groupby("gene", as_index=False)
        .agg(
            bootstrap_cdps_mean=("cdps_bootstrap", "mean"),
            bootstrap_cdps_ci_lower=("cdps_bootstrap", lambda s: float(np.quantile(s, 0.025))),
            bootstrap_cdps_ci_upper=("cdps_bootstrap", lambda s: float(np.quantile(s, 0.975))),
        )
    )
    updated = gene_summary.merge(stats_df, on="gene", how="left")
    return bootstrap_df, updated


def parse_gmt(path: str | None) -> dict[str, set[str]]:
    if not path or not Path(path).exists():
        return {}
    pathways: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathways[parts[0]] = {gene for gene in parts[2:] if gene}
    return pathways


def build_pathway_summary(ranked_genes: pd.DataFrame, config: Step5Config) -> pd.DataFrame:
    pathways = parse_gmt(config.pathway_gmt)
    if not pathways:
        return pd.DataFrame()
    top25 = set(ranked_genes.nsmallest(25, columns="cdps_rank")["gene"])
    top100 = set(ranked_genes.nsmallest(100, columns="cdps_rank")["gene"])
    rows: list[dict[str, Any]] = []
    gene_indexed = ranked_genes.set_index("gene")
    for pathway, genes in pathways.items():
        overlapping = sorted(set(genes) & set(gene_indexed.index))
        if not overlapping:
            continue
        sub = gene_indexed.loc[overlapping]
        rows.append(
            {
                "pathway": pathway,
                "pathway_gene_count": len(overlapping),
                "top25_hits": len(top25 & set(overlapping)),
                "top100_hits": len(top100 & set(overlapping)),
                "mean_cdps": float(sub["cdps_score"].mean()),
                "mean_attribution_score": float(sub["attribution_score"].mean()),
                "mean_perturbation_score": float(sub["perturbation_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["top25_hits", "mean_cdps", "top100_hits"], ascending=False)


def _label_top_points(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str, n: int = 10) -> None:
    for _, row in frame.head(n).iterrows():
        ax.text(row[x_col], row[y_col], str(row["gene"]), fontsize=9, fontweight="bold", family="serif")


def plot_top_cdps_genes(ranked: pd.DataFrame, output_dir: Path) -> None:
    palette = get_nature_palette()
    top = ranked.head(25).sort_values("cdps_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.barh(top["gene"], top["cdps_score"], color=palette["tumor"])
    ax.set_xlabel("Candidate Driver Priority Score (CDPS)")
    ax.set_ylabel("Gene")
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_cdps_top_genes.svg")
    plt.close(fig)


def plot_stability_heatmap(run_scores: pd.DataFrame, ranked: pd.DataFrame, output_dir: Path) -> None:
    if run_scores.empty:
        return
    top_genes = ranked.head(25)["gene"].tolist()
    heat = run_scores[run_scores["gene"].isin(top_genes)].copy()
    heat["run_id"] = heat["seed"].astype(str) + "_fold" + heat["outer_fold"].astype(str)
    pivot = heat.pivot_table(index="gene", columns="run_id", values="rank_within_run", aggfunc="mean")
    pivot = pivot.loc[[gene for gene in top_genes if gene in pivot.index]]
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(pivot))), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("Seed / outer fold run")
    ax.set_ylabel("Top CDPS genes")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Within-run rank", fontweight="bold", family="serif")
    apply_text_style(ax)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("serif")
    save_publication_figure(fig, output_dir / "figure6_stability_heatmap.svg")
    plt.close(fig)


def plot_attribution_vs_perturbation(ranked: pd.DataFrame, output_dir: Path) -> None:
    palette = get_nature_palette()
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(ranked["attribution_score"], ranked["perturbation_score"], color=palette["eac"], alpha=0.85, s=38)
    _label_top_points(ax, ranked, "attribution_score", "perturbation_score", n=12)
    ax.set_xlabel("Attribution score")
    ax.set_ylabel("Perturbation score")
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_attribution_vs_perturbation.svg")
    plt.close(fig)


def plot_cdps_components(ranked: pd.DataFrame, output_dir: Path) -> None:
    palette = get_nature_palette()
    top = ranked.head(20).sort_values("cdps_score", ascending=True)
    components = [
        ("attribution_score", palette["tumor"]),
        ("gate_score", palette["normal"]),
        ("stability_score", palette["smoker"]),
        ("invariance_score", palette["eac"]),
        ("perturbation_score", palette["escc"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    left = np.zeros(len(top))
    for column, color in components:
        ax.barh(top["gene"], top[column], left=left, color=color, label=column.replace("_", " "))
        left += top[column].to_numpy()
    ax.set_xlabel("Normalized CDPS components")
    ax.set_ylabel("Gene")
    ax.legend(loc="lower right")
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_cdps_components.svg")
    plt.close(fig)


def plot_environment_consistency(invariance_summary: pd.DataFrame, ranked: pd.DataFrame, output_dir: Path) -> None:
    if invariance_summary.empty:
        return
    palette = get_nature_palette()
    top_genes = ranked.head(20)["gene"].tolist()
    plot_df = invariance_summary[invariance_summary["gene"].isin(top_genes)].copy()
    plot_df = plot_df.sort_values(["environment_column", "invariance_score"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for env_column, subset in plot_df.groupby("environment_column"):
        ax.plot(subset["gene"], subset["invariance_score"], marker="o", label=env_column)
    ax.set_xlabel("Top CDPS genes")
    ax.set_ylabel("Environment consistency score")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(loc="best", ncol=2)
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_environment_consistency.svg")
    plt.close(fig)


def plot_counterfactual_perturbation(perturb_summary: pd.DataFrame, ranked: pd.DataFrame, output_dir: Path) -> None:
    if perturb_summary.empty:
        return
    palette = get_nature_palette()
    plot_df = ranked[["gene", "cdps_rank"]].merge(perturb_summary, on="gene", how="inner").sort_values("cdps_rank").head(20)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    y = np.arange(len(plot_df))
    ax.barh(y - 0.18, plot_df["mean_delta_prob_tumor"], height=0.35, color=palette["tumor"], label="Tumor→normal-like perturbation")
    ax.barh(y + 0.18, plot_df["mean_delta_prob_normal"], height=0.35, color=palette["normal"], label="Normal→tumor-like perturbation")
    ax.set_yticks(y, labels=plot_df["gene"])
    ax.set_xlabel("Mean Δ predicted tumor probability")
    ax.set_ylabel("Gene")
    ax.legend(loc="best")
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_counterfactual_perturbation.svg")
    plt.close(fig)


def plot_pathway_summary(pathway_summary: pd.DataFrame, output_dir: Path) -> None:
    if pathway_summary.empty:
        return
    palette = get_nature_palette()
    top = pathway_summary.head(15).sort_values("mean_cdps", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(top["pathway"], top["mean_cdps"], color=palette["eac"])
    ax.set_xlabel("Mean pathway CDPS")
    ax.set_ylabel("Pathway")
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_pathway_summary.svg")
    plt.close(fig)


def plot_latent_shift(latent_summary: pd.DataFrame, ranked: pd.DataFrame, output_dir: Path) -> None:
    if latent_summary.empty:
        return
    palette = get_nature_palette()
    plot_df = ranked[["gene", "cdps_rank"]].merge(latent_summary.groupby("gene", as_index=False)["latent_shift_norm"].mean(), on="gene", how="inner").sort_values("cdps_rank").head(20)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.barh(plot_df["gene"][::-1], plot_df["latent_shift_norm"][::-1], color=palette["smoker"])
    ax.set_xlabel("Mean latent shift norm")
    ax.set_ylabel("Gene")
    apply_text_style(ax)
    save_publication_figure(fig, output_dir / "figure6_latent_shift.svg")
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    attribution_summary: pd.DataFrame,
    samplewise_attributions: pd.DataFrame,
    gate_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    invariance_summary: pd.DataFrame,
    environment_effects: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    perturbation_samplewise: pd.DataFrame,
    ranked: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    latent_summary: pd.DataFrame,
    config: Step5Config,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution_summary.to_csv(output_dir / "gene_attribution_summary.csv", index=False)
    if config.save_samplewise_attributions and not samplewise_attributions.empty:
        samplewise_attributions.to_csv(output_dir / "samplewise_attributions.csv", index=False)
    gate_summary.to_csv(output_dir / "gate_importance_summary.csv", index=False)
    stability_summary.to_csv(output_dir / "gene_stability_summary.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "bootstrap_ranking_summary.csv", index=False)
    invariance_summary.to_csv(output_dir / "gene_invariance_summary.csv", index=False)
    environment_effects.to_csv(output_dir / "environment_gene_effects.csv", index=False)
    perturbation_summary.to_csv(output_dir / "gene_perturbation_summary.csv", index=False)
    perturbation_samplewise.to_csv(output_dir / "perturbation_samplewise.csv", index=False)
    ranked.to_csv(output_dir / "ranked_genes_cdps.csv", index=False)
    ranked.head(25).to_csv(output_dir / "top25_genes_cdps.csv", index=False)
    ranked.head(100).to_csv(output_dir / "top100_genes_cdps.csv", index=False)
    if not pathway_summary.empty:
        pathway_summary.to_csv(output_dir / "pathway_ranking_summary.csv", index=False)
    if not latent_summary.empty:
        latent_summary.to_csv(output_dir / "latent_perturbation_summary.csv", index=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)


def update_readme_step5() -> None:
    return None


def _merge_gene_components(
    gene_attr: pd.DataFrame,
    gate_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    invariance_summary: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
) -> pd.DataFrame:
    attr = gene_attr[["gene", "attribution_score", "signed_attribution_score"]].copy() if not gene_attr.empty else pd.DataFrame(columns=["gene", "attribution_score", "signed_attribution_score"])
    gate = gate_summary.rename(columns={"mean_abs_gate_weight": "gate_score"}).copy() if not gate_summary.empty else pd.DataFrame(columns=["gene", "gate_score"])
    stability = stability_summary.rename(columns={"stability_score_raw": "stability_score"}).copy() if not stability_summary.empty else pd.DataFrame(columns=["gene", "stability_score"])
    if not invariance_summary.empty:
        invariance = invariance_summary.groupby("gene", as_index=False).agg(
            invariance_score=("invariance_score", "mean"),
            environment_coverage=("environment_coverage", "sum"),
        )
    else:
        invariance = pd.DataFrame(columns=["gene", "invariance_score", "environment_coverage"])
    if perturbation_summary.empty:
        perturb = pd.DataFrame(columns=["gene", "perturbation_score", "directional_consistency", "mean_abs_delta_prob", "latent_shift_norm", "mean_delta_prob_tumor", "mean_delta_prob_normal"])
    else:
        perturb = perturbation_summary.copy()
    summary = attr.merge(gate, on="gene", how="outer")
    summary = summary.merge(stability, on="gene", how="outer")
    summary = summary.merge(invariance, on="gene", how="left")
    summary = summary.merge(perturb, on="gene", how="left")
    return summary


def main() -> None:
    args = parse_args()
    config = build_config(args)
    configure_publication_plotting()
    output_dir = Path(config.output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(42)

    print("[Step5] Loading inputs.")
    inputs = load_inputs(config)
    merged, normalized, deep_oof = align_inputs(inputs, config)
    model_artifacts = load_step4_models(config)

    print("[Step5] Aggregating gate evidence.")
    gate_per_run, gate_summary = aggregate_gate_weights(inputs["gate_weights"])

    all_attr_runs: list[pd.DataFrame] = []
    all_samplewise_attr: list[pd.DataFrame] = []
    if model_artifacts:
        print("[Step5] Computing fold-wise attributions from Step 4 checkpoints.")
        for artifact in model_artifacts:
            summary, samplewise = compute_attributions_for_artifact(artifact, merged, normalized, config)
            if not summary.empty:
                all_attr_runs.append(summary)
            if not samplewise.empty:
                all_samplewise_attr.append(samplewise)
    else:
        print("[Step5] No checkpoints available; attribution-dependent components will be reduced.")

    attribution_per_run = pd.concat(all_attr_runs, ignore_index=True) if all_attr_runs else pd.DataFrame()
    samplewise_attributions = pd.concat(all_samplewise_attr, ignore_index=True) if all_samplewise_attr else pd.DataFrame()
    attribution_summary, gene_attr = aggregate_attribution_scores(attribution_per_run)

    print("[Step5] Computing stability summaries.")
    run_scores = build_run_score_table(attribution_per_run, gate_per_run)
    stability_summary = compute_stability_summary(run_scores)

    preliminary = _merge_gene_components(gene_attr, gate_summary, stability_summary, pd.DataFrame(), pd.DataFrame())
    preliminary[["attribution_score", "gate_score", "stability_score"]] = preliminary[["attribution_score", "gate_score", "stability_score"]].fillna(0.0)
    preliminary["preliminary_rank_score"] = preliminary[["attribution_score", "gate_score", "stability_score"]].mean(axis=1)
    perturbation_genes = preliminary.sort_values("preliminary_rank_score", ascending=False)["gene"].head(config.top_genes_for_perturbation).tolist()

    print(f"[Step5] Running in silico perturbation for {len(perturbation_genes)} gene(s).")
    perturbation_summary, perturbation_samplewise, latent_summary = run_gene_perturbation(perturbation_genes, model_artifacts, merged, normalized, config)

    print("[Step5] Quantifying environment consistency.")
    invariance_summary, environment_effects, _ = compute_environment_consistency(samplewise_attributions, merged, perturbation_samplewise, config)

    gene_summary = _merge_gene_components(gene_attr, gate_summary, stability_summary, invariance_summary, perturbation_summary)
    gene_summary["mean_abs_attribution"] = gene_summary["attribution_score"].fillna(0.0)
    gene_summary["perturbation_directional_consistency"] = gene_summary.get("directional_consistency", pd.Series(dtype=float)).fillna(0.0)
    gene_summary = compute_cdps(gene_summary, config)

    print("[Step5] Bootstrapping CDPS rankings.")
    bootstrap_summary, gene_summary = bootstrap_cdps(run_scores, gene_summary, config)
    ranked = rank_genes(gene_summary)

    pathway_summary = build_pathway_summary(ranked, config) if (config.run_pathway_summary or config.pathway_gmt) else pd.DataFrame()

    print("[Step5] Saving tables and publication-grade SVG figures.")
    save_outputs(
        output_dir,
        attribution_summary,
        samplewise_attributions,
        gate_summary,
        gene_summary[[
            "gene",
            "top25_frequency",
            "top100_frequency",
            "mean_rank",
            "rank_std",
            "bootstrap_cdps_mean",
            "bootstrap_cdps_ci_lower",
            "bootstrap_cdps_ci_upper",
        ]].copy(),
        bootstrap_summary,
        invariance_summary,
        environment_effects,
        perturbation_summary,
        perturbation_samplewise,
        ranked[[
            "gene",
            "cdps_score",
            "cdps_rank",
            "attribution_score",
            "gate_score",
            "stability_score",
            "invariance_score",
            "perturbation_score",
            "mean_gate_weight",
            "gate_selection_frequency",
            "mean_abs_attribution",
            "top25_frequency",
            "top100_frequency",
            "perturbation_directional_consistency",
            "notes",
        ] + [c for c in ["bootstrap_cdps_mean", "bootstrap_cdps_ci_lower", "bootstrap_cdps_ci_upper"] if c in ranked.columns]].copy(),
        pathway_summary,
        latent_summary,
        config,
    )

    plot_top_cdps_genes(ranked, figure_dir)
    plot_stability_heatmap(run_scores, ranked, figure_dir)
    plot_attribution_vs_perturbation(ranked, figure_dir)
    plot_cdps_components(ranked, figure_dir)
    plot_environment_consistency(invariance_summary, ranked, figure_dir)
    plot_counterfactual_perturbation(perturbation_summary, ranked, figure_dir)
    plot_pathway_summary(pathway_summary, figure_dir)
    plot_latent_shift(latent_summary, ranked, figure_dir)

    print(
        "[Step5] Completed CDPS prioritization. "
        f"Ranked genes: {len(ranked)} | Attributed genes: {attribution_summary['gene'].nunique() if not attribution_summary.empty else 0} | "
        f"Perturbed genes: {perturbation_summary['gene'].nunique() if not perturbation_summary.empty else 0}."
    )


if __name__ == "__main__":
    main()
