#!/usr/bin/env python3
"""Step 12: package a reproducibility and release bundle for the study.

This script is a packaging-only continuation step. It validates outputs from prior
pipeline stages, copies eligible artifacts into a deterministic release structure,
generates a manifest/checksums/environment snapshot, writes release documentation,
and optionally creates distribution archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

SCRIPT_NAME = "12_package_reproducibility_and_release"
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_CHECKSUM_ALGORITHM = "sha256"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RELEASE_ARCHIVE_STEM = "kinase_causality_qsar_release"


class Step12ConfigurationError(ValueError):
    """Raised when required configuration values are missing or invalid."""


class Step12ValidationError(RuntimeError):
    """Raised when required package assets are missing or internally inconsistent."""


@dataclass(frozen=True)
class CategorySpec:
    config_flag: str
    source: str
    destination: str
    asset_category: str
    required_if_enabled: bool = False


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec("include_raw_data", "data/raw", "data/raw", "data_raw"),
    CategorySpec("include_interim_data", "data/interim", "data/interim", "data_interim"),
    CategorySpec("include_processed_data", "data/processed", "data/processed", "data_processed"),
    CategorySpec("include_processed_data", "data/splits", "data/splits", "splits"),
    CategorySpec("include_models", "models", "models", "model"),
    CategorySpec("include_results", "results", "results", "result"),
    CategorySpec("include_figures", "figures", "figures", "figure"),
    CategorySpec("include_manuscript_outputs", "manuscript_outputs", "manuscript_outputs", "manuscript_asset"),
    CategorySpec("include_reports", "reports", "reports", "report"),
    CategorySpec("include_logs", "logs", "logs", "log"),
    CategorySpec("include_configs_used", "configs_used", "configs_used", "config_snapshot"),
)

CONFIG_DEFAULTS: dict[str, Any] = {
    "project_root": ".",
    "output_release_root": "release_package",
    "output_archive_root": "release_archives",
    "output_manifest_path": "release_package/release_manifest.csv",
    "output_report_path": "reports/12_reproducibility_and_release_report.json",
    "output_release_readme_path": "release_package/README_RELEASE.md",
    "output_runbook_path": "release_package/RUNBOOK.md",
    "output_environment_snapshot_path": "release_package/environment_snapshot.txt",
    "output_directory_tree_path": "release_package/directory_tree.txt",
    "include_raw_data": True,
    "include_interim_data": True,
    "include_processed_data": True,
    "include_models": True,
    "include_results": True,
    "include_figures": True,
    "include_manuscript_outputs": True,
    "include_reports": True,
    "include_logs": True,
    "include_configs_used": True,
    "validate_required_assets": True,
    "generate_checksums": True,
    "checksum_algorithm": DEFAULT_CHECKSUM_ALGORITHM,
    "create_tar_gz": True,
    "create_zip": False,
    "required_assets": [
        "manuscript_outputs/manuscript_asset_manifest.csv",
        "reports/10_model_comparison_and_interpretation_report.json",
        "reports/11_manuscript_figures_and_tables_report.json",
        "README.md",
    ],
    "optional_environment_files": [
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
        "setup.cfg",
    ],
    "save_config_snapshot": True,
}

MANIFEST_COLUMNS = [
    "asset_id",
    "relative_path",
    "absolute_source_path",
    "asset_category",
    "file_type",
    "file_size_bytes",
    "checksum",
    "originating_step",
    "required_or_optional",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the root config YAML file.")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise Step12ConfigurationError(f"Config file must define a top-level mapping: {path}")
    return data


def resolve_script12_config(config_path: Path) -> dict[str, Any]:
    raw_config = load_yaml(config_path)
    if "script_12" not in raw_config or not isinstance(raw_config["script_12"], dict):
        raise Step12ConfigurationError(
            "Missing required 'script_12' section in config.yaml. Please add the Step-12 packaging configuration."
        )
    merged = dict(CONFIG_DEFAULTS)
    merged.update(raw_config["script_12"])
    missing = [key for key in [
        "project_root",
        "output_release_root",
        "output_archive_root",
        "output_manifest_path",
        "output_report_path",
        "output_release_readme_path",
        "output_runbook_path",
        "output_environment_snapshot_path",
        "output_directory_tree_path",
    ] if not merged.get(key)]
    if missing:
        raise Step12ConfigurationError(
            f"Missing required script_12 config values: {', '.join(sorted(missing))}"
        )
    return merged


def setup_logging(project_root: Path) -> tuple[logging.Logger, Path]:
    log_dir = project_root / "logs"
    ensure_directory(log_dir)
    timestamp = utc_now().strftime(TIMESTAMP_FORMAT)
    log_path = log_dir / f"{SCRIPT_NAME}_{timestamp}.log"

    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger, log_path


def clean_release_root(release_root: Path, logger: logging.Logger) -> None:
    if release_root.exists():
        logger.info("Removing previous release directory to keep packaging deterministic: %s", release_root)
        shutil.rmtree(release_root)
    ensure_directory(release_root)


def snapshot_config(project_root: Path, raw_root_config: dict[str, Any], logger: logging.Logger) -> Path:
    configs_dir = project_root / "configs_used"
    ensure_directory(configs_dir)
    destination = configs_dir / f"{SCRIPT_NAME}_config.yaml"
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_root_config, handle, sort_keys=False)
    logger.info("Saved config snapshot to %s", destination)
    return destination


def validate_required_assets(project_root: Path, required_assets: Iterable[str], logger: logging.Logger) -> list[str]:
    missing: list[str] = []
    for asset in sorted(set(required_assets)):
        candidate = project_root / asset
        if candidate.exists():
            logger.info("Validated required asset: %s", candidate)
        else:
            logger.error("Missing required asset: %s", candidate)
            missing.append(asset)
    return missing


def checksum_for_file(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def infer_originating_step(path: Path) -> str:
    normalized_parts = [part.lower() for part in path.parts]
    for index in range(1, 13):
        script_token = f"script_{index:02d}"
        step_token = f"step{index}"
        step_token_underscore = f"step_{index}"
        if any(token in part for part in normalized_parts for token in (script_token, step_token, step_token_underscore)):
            return script_token
        if any(part.startswith(f"{index:02d}_") for part in normalized_parts):
            return script_token
    return "external_or_project_root"


def asset_file_type(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    return suffixes.lstrip(".") or "no_extension"


def register_release_file(
    source_file: Path,
    release_relative_str: str,
    packaged_file: Path,
    asset_category: str,
    required_or_optional: str,
    notes: str,
    manifest_rows: list[dict[str, Any]],
    seen_release_paths: set[str],
    logger: logging.Logger,
) -> None:
    if release_relative_str in seen_release_paths:
        raise Step12ValidationError(f"Duplicate relative path created in release package: {release_relative_str}")
    seen_release_paths.add(release_relative_str)
    logger.info("Registered packaged file %s", packaged_file)
    manifest_rows.append(
        {
            "asset_id": f"asset_{len(manifest_rows) + 1:05d}",
            "relative_path": release_relative_str,
            "absolute_source_path": str(source_file.resolve()),
            "asset_category": asset_category,
            "file_type": asset_file_type(packaged_file),
            "file_size_bytes": packaged_file.stat().st_size,
            "checksum": "",
            "originating_step": infer_originating_step(source_file),
            "required_or_optional": required_or_optional,
            "notes": notes,
        }
    )


def copy_file_record(
    source_file: Path,
    destination_file: Path,
    asset_category: str,
    required_or_optional: str,
    notes: str,
    manifest_rows: list[dict[str, Any]],
    seen_release_paths: set[str],
    logger: logging.Logger,
) -> None:
    if destination_file.exists():
        raise Step12ValidationError(f"Duplicate destination detected before copy: {destination_file}")
    ensure_directory(destination_file.parent)
    release_relative_str = notes.split("release_relative=", 1)[1].split(";", 1)[0] if "release_relative=" in notes else destination_file.name
    shutil.copy2(source_file, destination_file)
    logger.info("Packaged %s -> %s", source_file, destination_file)
    register_release_file(
        source_file=source_file,
        release_relative_str=release_relative_str,
        packaged_file=destination_file,
        asset_category=asset_category,
        required_or_optional=required_or_optional,
        notes=notes,
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        logger=logger,
    )


def package_category(
    project_root: Path,
    release_root: Path,
    spec: CategorySpec,
    include_flag: bool,
    manifest_rows: list[dict[str, Any]],
    seen_release_paths: set[str],
    warnings: list[str],
    logger: logging.Logger,
) -> None:
    if not include_flag:
        logger.info("Skipping category %s because config flag %s is disabled.", spec.asset_category, spec.config_flag)
        return
    source_root = project_root / spec.source
    if not source_root.exists():
        warning = f"Requested category '{spec.asset_category}' not found at {source_root}; category skipped."
        warnings.append(warning)
        logger.warning(warning)
        return
    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not source_files:
        warning = f"Requested category '{spec.asset_category}' exists but contains no files: {source_root}"
        warnings.append(warning)
        logger.warning(warning)
        return
    logger.info("Packaging category %s from %s with %d files", spec.asset_category, source_root, len(source_files))
    for source_file in source_files:
        rel_from_source = source_file.relative_to(source_root)
        destination = release_root / spec.destination / rel_from_source
        release_relative = destination.relative_to(release_root).as_posix()
        notes = f"release_relative={release_relative}; copied_from={spec.source}"
        copy_file_record(
            source_file=source_file,
            destination_file=destination,
            asset_category=spec.asset_category,
            required_or_optional="optional",
            notes=notes,
            manifest_rows=manifest_rows,
            seen_release_paths=seen_release_paths,
            logger=logger,
        )


def package_single_file(
    source_file: Path,
    release_root: Path,
    destination_relative: str,
    asset_category: str,
    required_or_optional: str,
    notes: str,
    manifest_rows: list[dict[str, Any]],
    seen_release_paths: set[str],
    logger: logging.Logger,
) -> None:
    destination = release_root / destination_relative
    notes_with_rel = f"release_relative={destination_relative}; {notes}" if notes else f"release_relative={destination_relative}"
    if source_file.resolve() == destination.resolve():
        if not destination.exists():
            raise Step12ValidationError(f"Expected generated release file does not exist: {destination}")
        register_release_file(
            source_file=source_file,
            release_relative_str=destination_relative,
            packaged_file=destination,
            asset_category=asset_category,
            required_or_optional=required_or_optional,
            notes=notes_with_rel,
            manifest_rows=manifest_rows,
            seen_release_paths=seen_release_paths,
            logger=logger,
        )
        return
    copy_file_record(
        source_file=source_file,
        destination_file=destination,
        asset_category=asset_category,
        required_or_optional=required_or_optional,
        notes=notes_with_rel,
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        logger=logger,
    )


def package_environment_files(
    project_root: Path,
    release_root: Path,
    optional_environment_files: Iterable[str],
    manifest_rows: list[dict[str, Any]],
    seen_release_paths: set[str],
    missing_optional_assets: list[str],
    logger: logging.Logger,
) -> None:
    for relative_path in optional_environment_files:
        source = project_root / relative_path
        if source.exists() and source.is_file():
            package_single_file(
                source_file=source,
                release_root=release_root,
                destination_relative=Path(relative_path).name,
                asset_category="environment_file",
                required_or_optional="optional",
                notes="optional environment/dependency file",
                manifest_rows=manifest_rows,
                seen_release_paths=seen_release_paths,
                logger=logger,
            )
        else:
            missing_optional_assets.append(relative_path)
            logger.warning("Optional environment file not found: %s", source)


def package_root_readme(
    project_root: Path,
    release_root: Path,
    manifest_rows: list[dict[str, Any]],
    seen_release_paths: set[str],
    logger: logging.Logger,
) -> None:
    source = project_root / "README.md"
    if source.exists():
        package_single_file(
            source_file=source,
            release_root=release_root,
            destination_relative="README_PROJECT.md",
            asset_category="readme",
            required_or_optional="required",
            notes="project root README snapshot",
            manifest_rows=manifest_rows,
            seen_release_paths=seen_release_paths,
            logger=logger,
        )


def collect_package_counts(manifest_rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row["asset_category"] for row in manifest_rows)
    return dict(sorted(counter.items()))


def write_manifest(manifest_rows: list[dict[str, Any]], manifest_path: Path, logger: logging.Logger) -> None:
    ordered_rows = sorted(manifest_rows, key=lambda row: (row["relative_path"], row["asset_category"], row["asset_id"]))
    for index, row in enumerate(ordered_rows, start=1):
        row["asset_id"] = f"asset_{index:05d}"
    ensure_directory(manifest_path.parent)
    manifest_df = pd.DataFrame(ordered_rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Wrote release manifest with %d rows to %s", len(ordered_rows), manifest_path)


def generate_checksums(
    release_root: Path,
    manifest_rows: list[dict[str, Any]],
    algorithm: str,
    logger: logging.Logger,
) -> Path:
    checksum_path = release_root / "checksums.txt"
    lines: list[str] = []
    checksum_by_relative_path: dict[str, str] = {}
    for row in sorted(manifest_rows, key=lambda item: item["relative_path"]):
        packaged_file = release_root / row["relative_path"]
        if not packaged_file.exists():
            raise Step12ValidationError(f"Cannot checksum missing packaged file: {packaged_file}")
        checksum = checksum_for_file(packaged_file, algorithm)
        row["checksum"] = checksum
        checksum_by_relative_path[row["relative_path"]] = checksum
        lines.append(f"{checksum}  {row['relative_path']}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("Generated %s checksums at %s", algorithm, checksum_path)
    return checksum_path


def library_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "not installed or unavailable"


def write_environment_snapshot(
    project_root: Path,
    output_path: Path,
    optional_environment_files: Iterable[str],
    logger: logging.Logger,
) -> None:
    package_map = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "scikit-learn",
        "rdkit": "rdkit",
        "torch": "torch",
        "torch_geometric": "torch-geometric",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "matplotlib": "matplotlib",
        "pyyaml": "PyYAML",
    }
    lines = [
        f"generated_utc: {utc_now().isoformat()}",
        f"python_version: {sys.version.replace(os.linesep, ' ')}",
        f"python_executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"system: {platform.system()}",
        f"release: {platform.release()}",
        f"machine: {platform.machine()}",
        f"cwd: {Path.cwd().resolve()}",
        "",
        "library_versions:",
    ]
    for label, package_name in package_map.items():
        lines.append(f"- {label}: {library_version(package_name)}")
    lines.extend(["", "environment_files_present:"])
    for relative_path in optional_environment_files:
        lines.append(f"- {relative_path}: {'present' if (project_root / relative_path).exists() else 'missing'}")
    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote environment snapshot to %s", output_path)


def build_directory_tree_lines(root: Path) -> list[str]:
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str) -> None:
        entries = sorted(directory.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
        for index, entry in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "│   "
                walk(entry, prefix + extension)

    walk(root, "")
    return lines


def write_directory_tree(release_root: Path, output_path: Path, logger: logging.Logger) -> None:
    tree_lines = build_directory_tree_lines(release_root)
    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    logger.info("Wrote directory tree snapshot to %s", output_path)


def write_release_readme(release_root: Path, output_path: Path, packaged_categories: list[str], logger: logging.Logger) -> None:
    content = f"""# Release Package

This release bundle is the final reproducibility and archival package assembled by Step-12 for the kinase causality QSAR study. It packages validated outputs from earlier steps without retraining models or recomputing benchmark results.

## What this package contains

- Release documentation for inspection and handoff.
- A machine-readable asset manifest (`release_manifest.csv`).
- Checksums for packaged files when enabled (`checksums.txt`).
- Environment and dependency snapshot details (`environment_snapshot.txt`).
- A deterministic directory-tree summary (`directory_tree.txt`).
- Copied study artifacts from requested pipeline categories: {', '.join(packaged_categories) if packaged_categories else 'none packaged'}.

## Major directories

- `data/`: raw, interim, processed, and split-level data snapshots when present.
- `models/`: trained model artifacts copied from previous modeling steps.
- `results/`: benchmarking, comparison, and integration results.
- `figures/`: packaged figure outputs from prior steps.
- `manuscript_outputs/`: main/supplementary figures, tables, and source-data assets used for manuscript support.
- `reports/`: JSON or text reports describing earlier stages and this packaging step.
- `logs/`: pipeline log files including the Step-12 packaging log.
- `configs_used/`: config snapshots and provenance-supporting configuration files.

## How to inspect key outputs

- Manuscript-facing assets: review `manuscript_outputs/` and `README_PROJECT.md`.
- Model comparison outputs: inspect `results/` and `reports/10_model_comparison_and_interpretation_report.json` if present.
- Final figures and tables: inspect `figures/`, `manuscript_outputs/`, and `reports/11_manuscript_figures_and_tables_report.json` if present.
- Source data for figures/tables: inspect the relevant subfolders beneath `manuscript_outputs/`.

## Provenance note

This package is a release bundle assembled from prior pipeline steps. Step-12 validates and packages artifacts; it does not rerun modeling, regenerate scientific findings, or overwrite source study outputs.
"""
    ensure_directory(output_path.parent)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote release README to %s", output_path)


def write_runbook(output_path: Path, logger: logging.Logger) -> None:
    content = """# Step-12 Runbook

## Expected pipeline order

1. Step-01: data assembly / preprocessing
2. Step-02: task construction / annotation refinement
3. Step-03: split generation
4. Step-04: baseline or supporting model preparation
5. Step-05: additional modeling and analysis outputs
6. Step-06: validated processed study assets
7. Step-07: baseline model training artifacts
8. Step-08: causal model training artifacts
9. Step-09: benchmark and causal-QSAR result collection
10. Step-10: model comparison and interpretation integration
11. Step-11: manuscript-grade figures and tables
12. Step-12: reproducibility, release, and archival packaging

## Folder-to-stage mapping

- `data/`: inputs, curated datasets, processed matrices, and split files from earlier preparation steps.
- `models/`: trained model artifacts saved by modeling stages.
- `results/`: score tables, benchmark summaries, and integration outputs.
- `figures/` and `manuscript_outputs/`: manuscript-facing visual and tabular assets.
- `reports/`: machine-readable stage summaries.
- `configs_used/`: exact config snapshots needed for provenance.

## Finding final assets

- Final manuscript assets: `manuscript_outputs/`
- Trained models: `models/`
- Benchmarking outputs: `results/`
- Config snapshots: `configs_used/`
- Packaging manifest and checksums: `release_manifest.csv` and `checksums.txt`

## Validation workflow

1. Confirm the files listed in `release_manifest.csv` exist in the release directory.
2. Recompute hashes and compare to `checksums.txt` when checksums are enabled.
3. Review `environment_snapshot.txt` for the captured Python/platform/package context.
4. Review `reports/12_reproducibility_and_release_report.json` for the packaging summary.

## Reproducibility notes

- This step is packaging-only and should not retrain or reevaluate models.
- Deterministic ordering is used for manifest rows, checksums, and directory-tree output.
- Source study artifacts remain untouched; only copied release assets are written into the release bundle.
"""
    ensure_directory(output_path.parent)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote runbook to %s", output_path)


def validate_release_outputs(
    release_root: Path,
    manifest_rows: list[dict[str, Any]],
    include_manuscript_outputs: bool,
    include_results: bool,
    generate_checksums_enabled: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    relative_paths = [row["relative_path"] for row in manifest_rows]
    duplicates = sorted({path for path in relative_paths if relative_paths.count(path) > 1})
    if duplicates:
        raise Step12ValidationError(f"Duplicate relative paths detected in release manifest: {duplicates}")
    missing_manifest_targets = [row["relative_path"] for row in manifest_rows if not (release_root / row["relative_path"]).exists()]
    if missing_manifest_targets:
        raise Step12ValidationError(
            "Manifest entries point to missing packaged files: " + ", ".join(missing_manifest_targets)
        )
    if include_manuscript_outputs and not (release_root / "manuscript_outputs/manuscript_asset_manifest.csv").exists():
        raise Step12ValidationError("Expected manuscript asset manifest inside release package but it is missing.")
    if include_results:
        required_result_reports = [
            release_root / "reports/10_model_comparison_and_interpretation_report.json",
            release_root / "reports/11_manuscript_figures_and_tables_report.json",
        ]
        missing_result_reports = [str(path.relative_to(release_root)) for path in required_result_reports if not path.exists()]
        if missing_result_reports:
            raise Step12ValidationError(
                "Expected packaged key reports are missing: " + ", ".join(missing_result_reports)
            )
    checksum_missing = []
    if generate_checksums_enabled:
        checksum_file = release_root / "checksums.txt"
        if not checksum_file.exists():
            raise Step12ValidationError("Checksum generation enabled but checksums.txt is missing.")
        for row in manifest_rows:
            if not row.get("checksum"):
                checksum_missing.append(row["relative_path"])
    if checksum_missing:
        raise Step12ValidationError(
            "Checksum generation enabled but some manifest entries do not contain checksums: " + ", ".join(checksum_missing)
        )
    logger.info("Release package validation completed successfully.")
    return {
        "manifest_entries_validated": len(manifest_rows),
        "duplicate_relative_paths": duplicates,
        "checksum_entries_missing": checksum_missing,
    }


def create_archives(release_root: Path, archive_root: Path, create_tar_gz: bool, create_zip: bool, logger: logging.Logger) -> list[str]:
    ensure_directory(archive_root)
    archive_paths: list[str] = []
    if create_tar_gz:
        tar_path = archive_root / f"{RELEASE_ARCHIVE_STEM}.tar.gz"
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as handle:
            handle.add(release_root, arcname=release_root.name)
        archive_paths.append(str(tar_path))
        logger.info("Created tar.gz archive at %s", tar_path)
    if create_zip:
        zip_path = archive_root / f"{RELEASE_ARCHIVE_STEM}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(release_root.rglob("*")):
                archive.write(path, arcname=Path(release_root.name) / path.relative_to(release_root))
        archive_paths.append(str(zip_path))
        logger.info("Created zip archive at %s", zip_path)
    return archive_paths


def write_report(report: dict[str, Any], output_path: Path, logger: logging.Logger) -> None:
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote Step-12 JSON report to %s", output_path)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw_root_config = load_yaml(config_path)
    script_config = resolve_script12_config(config_path)
    project_root = (config_path.parent / script_config["project_root"]).resolve()
    logger, log_path = setup_logging(project_root)
    logger.info("Starting %s", SCRIPT_NAME)
    logger.info("Using project root: %s", project_root)

    release_root = (project_root / script_config["output_release_root"]).resolve()
    archive_root = (project_root / script_config["output_archive_root"]).resolve()
    manifest_path = (project_root / script_config["output_manifest_path"]).resolve()
    report_path = (project_root / script_config["output_report_path"]).resolve()
    release_readme_path = (project_root / script_config["output_release_readme_path"]).resolve()
    runbook_path = (project_root / script_config["output_runbook_path"]).resolve()
    environment_snapshot_path = (project_root / script_config["output_environment_snapshot_path"]).resolve()
    directory_tree_path = (project_root / script_config["output_directory_tree_path"]).resolve()

    clean_release_root(release_root, logger)

    config_snapshot_path = snapshot_config(project_root, raw_root_config, logger) if script_config["save_config_snapshot"] else Path("")

    missing_required_assets = []
    if script_config["validate_required_assets"]:
        missing_required_assets = validate_required_assets(project_root, script_config["required_assets"], logger)
        if missing_required_assets:
            raise Step12ValidationError(
                "Step-12 packaging cannot proceed because required assets are missing: " + ", ".join(missing_required_assets)
            )

    manifest_rows: list[dict[str, Any]] = []
    seen_release_paths: set[str] = set()
    warnings: list[str] = []
    missing_optional_assets: list[str] = []

    packaged_categories: list[str] = []
    for spec in CATEGORY_SPECS:
        include_flag = bool(script_config[spec.config_flag])
        pre_count = len(manifest_rows)
        package_category(
            project_root=project_root,
            release_root=release_root,
            spec=spec,
            include_flag=include_flag,
            manifest_rows=manifest_rows,
            seen_release_paths=seen_release_paths,
            warnings=warnings,
            logger=logger,
        )
        if len(manifest_rows) > pre_count and spec.asset_category not in packaged_categories:
            packaged_categories.append(spec.asset_category)

    package_root_readme(project_root, release_root, manifest_rows, seen_release_paths, logger)
    packaged_categories.append("readme")

    package_environment_files(
        project_root=project_root,
        release_root=release_root,
        optional_environment_files=script_config["optional_environment_files"],
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        missing_optional_assets=missing_optional_assets,
        logger=logger,
    )
    if any(row["asset_category"] == "environment_file" for row in manifest_rows):
        packaged_categories.append("environment_file")

    if config_snapshot_path:
        config_snapshot_release_relative = f"configs_used/{config_snapshot_path.name}"
        if config_snapshot_release_relative not in seen_release_paths:
            package_single_file(
                source_file=config_snapshot_path,
                release_root=release_root,
                destination_relative=config_snapshot_release_relative,
                asset_category="config_snapshot",
                required_or_optional="required",
                notes="Step-12 config snapshot used for packaging",
                manifest_rows=manifest_rows,
                seen_release_paths=seen_release_paths,
                logger=logger,
            )
        if "config_snapshot" not in packaged_categories:
            packaged_categories.append("config_snapshot")

    write_environment_snapshot(project_root, environment_snapshot_path, script_config["optional_environment_files"], logger)
    package_single_file(
        source_file=environment_snapshot_path,
        release_root=release_root,
        destination_relative="environment_snapshot.txt",
        asset_category="environment_file",
        required_or_optional="required",
        notes="generated environment snapshot",
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        logger=logger,
    )

    write_release_readme(release_root, release_readme_path, sorted(set(packaged_categories)), logger)
    write_runbook(runbook_path, logger)

    package_single_file(
        source_file=release_readme_path,
        release_root=release_root,
        destination_relative="README_RELEASE.md",
        asset_category="readme",
        required_or_optional="required",
        notes="generated release README",
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        logger=logger,
    )
    package_single_file(
        source_file=runbook_path,
        release_root=release_root,
        destination_relative="RUNBOOK.md",
        asset_category="readme",
        required_or_optional="required",
        notes="generated Step-12 runbook",
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        logger=logger,
    )

    write_directory_tree(release_root, directory_tree_path, logger)
    package_single_file(
        source_file=directory_tree_path,
        release_root=release_root,
        destination_relative="directory_tree.txt",
        asset_category="report",
        required_or_optional="required",
        notes="generated release directory tree",
        manifest_rows=manifest_rows,
        seen_release_paths=seen_release_paths,
        logger=logger,
    )

    if script_config["generate_checksums"]:
        generate_checksums(release_root, manifest_rows, script_config["checksum_algorithm"], logger)

    write_manifest(manifest_rows, manifest_path, logger)

    validation_summary = validate_release_outputs(
        release_root=release_root,
        manifest_rows=manifest_rows,
        include_manuscript_outputs=bool(script_config["include_manuscript_outputs"]),
        include_results=bool(script_config["include_results"]),
        generate_checksums_enabled=bool(script_config["generate_checksums"]),
        logger=logger,
    )

    archive_paths = create_archives(
        release_root=release_root,
        archive_root=archive_root,
        create_tar_gz=bool(script_config["create_tar_gz"]),
        create_zip=bool(script_config["create_zip"]),
        logger=logger,
    )

    total_size = sum((release_root / row["relative_path"]).stat().st_size for row in manifest_rows)
    report = {
        "project_root_used": str(project_root),
        "release_root_path": str(release_root),
        "archive_paths": archive_paths,
        "packaging_categories_included": sorted(set(packaged_categories)),
        "total_packaged_file_count": len(manifest_rows),
        "total_packaged_size_bytes": total_size,
        "counts_by_asset_category": collect_package_counts(manifest_rows),
        "required_assets_validated": sorted(set(script_config["required_assets"])) if not missing_required_assets else [],
        "missing_optional_assets": sorted(set(missing_optional_assets)),
        "checksum_status": {
            "enabled": bool(script_config["generate_checksums"]),
            "algorithm": script_config["checksum_algorithm"],
            "path": str(release_root / "checksums.txt") if script_config["generate_checksums"] else None,
        },
        "environment_snapshot_status": {
            "path": str(environment_snapshot_path),
            "exists": environment_snapshot_path.exists(),
        },
        "archive_creation_status": {
            "create_tar_gz": bool(script_config["create_tar_gz"]),
            "create_zip": bool(script_config["create_zip"]),
            "archive_count": len(archive_paths),
        },
        "validation_summary": validation_summary,
        "warnings": warnings,
        "timestamp": utc_now().isoformat(),
        "config_snapshot_reference": str(config_snapshot_path) if config_snapshot_path else None,
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
    }
    write_report(report, report_path, logger)
    logger.info("Step-12 packaging completed successfully.")


if __name__ == "__main__":
    main()
