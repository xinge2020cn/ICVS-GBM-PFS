"""Command-line entry points for the complete research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .clinical import fit_clinical_model
from .config import StudyConfig, load_config
from .data import PROCESSED_IMAGE_COLUMNS, RAW_IMAGE_COLUMNS, read_manifest, validate_manifest
from .evaluation import evaluate_models
from .explain import exact_time_dependent_shapley, select_explanation_patients
from .icvs import fit_icvs_model
from .preprocessing import preprocess_manifest
from .radiomics import extract_radiomics_features, fit_radiomics_model
from .segmentation import (
    evaluate_segmentation_manifest,
    prepare_nnunet_dataset,
    run_nnunet_prediction,
    run_nnunet_training,
)
from .training import run_crossfit_and_refit, select_training_duration


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def _load_manifest_and_config(args: argparse.Namespace) -> tuple[pd.DataFrame, StudyConfig]:
    config = load_config(args.config)
    frame = read_manifest(args.manifest)
    return frame, config


def _write_table(table: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)


def command_validate(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    path_columns: tuple[str, ...]
    if args.paths == "raw":
        path_columns = RAW_IMAGE_COLUMNS
    elif args.paths == "processed":
        path_columns = PROCESSED_IMAGE_COLUMNS
    else:
        path_columns = ()
    audit = validate_manifest(frame, config, require_paths=path_columns)
    print(
        json.dumps(
            {
                "patients": audit.patients,
                "cohorts": audit.cohorts,
                "centers": audit.centers,
                "biological_subset": audit.biological_subset,
            },
            indent=2,
        )
    )


def command_preprocess(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    preprocess_manifest(
        args.manifest,
        config,
        args.output_root,
        args.output_manifest,
    )


def command_prepare_nnunet(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config, require_paths=PROCESSED_IMAGE_COLUMNS)
    path = prepare_nnunet_dataset(frame, config, args.nnunet_raw)
    print(path)


def command_train_nnunet(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_nnunet_training(
        config,
        raw=args.nnunet_raw,
        preprocessed=args.nnunet_preprocessed,
        results=args.nnunet_results,
        trainer=args.trainer,
    )


def command_predict_nnunet(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_nnunet_prediction(
        config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        raw=args.nnunet_raw,
        preprocessed=args.nnunet_preprocessed,
        results=args.nnunet_results,
        trainer=args.trainer,
    )


def command_evaluate_segmentation(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    source_root = Path(args.manifest).resolve().parent
    for column in (args.reference_column, args.prediction_column):
        if column not in frame:
            raise ValueError(f"Segmentation manifest is missing column: {column}")
        frame[column] = frame[column].map(
            lambda value: str(
                (source_root / str(value)).resolve()
                if not Path(str(value)).is_absolute()
                else Path(str(value)).resolve()
            )
        )
    patient, summary = evaluate_segmentation_manifest(
        frame,
        config,
        reference_column=args.reference_column,
        prediction_column=args.prediction_column,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=config.seed,
    )
    output = Path(args.output_dir).resolve()
    _write_table(patient, output / "segmentation_patient_metrics.csv")
    _write_table(summary, output / "segmentation_summary.csv")


def command_extract_radiomics(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config, require_paths=PROCESSED_IMAGE_COLUMNS)
    result = extract_radiomics_features(frame, config, args.parameters)
    _write_table(result, args.output)


def command_train_radiomics(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config)
    features = pd.read_csv(args.features)
    fit_radiomics_model(frame, features, config, args.output_dir)


def command_fit_clinical(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config)
    fit_clinical_model(frame, config, args.output_dir)


def command_select_duration(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config, require_paths=PROCESSED_IMAGE_COLUMNS)
    training = frame.loc[frame[config.column("cohort")].eq(config.cohort("training"))].reset_index(
        drop=True
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = select_training_duration(
        training,
        config,
        model_name=args.model,
        device=device,
        cache_dir=args.cache_dir,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs_completed,
                "history": result.history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def command_train_deep(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config, require_paths=PROCESSED_IMAGE_COLUMNS)
    run_crossfit_and_refit(
        frame,
        config,
        model_name=args.model,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device_name=args.device,
        epochs_override=args.epochs,
    )


def command_fit_icvs(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config)
    scores = pd.read_csv(args.vit_scores)
    fit_icvs_model(frame, scores, config, args.output_dir)


def _standard_prediction_table(
    manifest: pd.DataFrame,
    config: StudyConfig,
    path: Path,
    *,
    model: str,
    prefix: str,
    deep: bool,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    horizons = config.section("icvs")["horizons_months"]
    table = manifest[[patient_col, cohort_col]].merge(
        table, on=[patient_col, cohort_col], how="left", validate="one_to_one"
    )
    if deep:
        training = table[cohort_col].eq(config.cohort("training"))
        risk = table[f"{prefix}_score_final"].to_numpy(float)
        risk[training] = table.loc[training, f"{prefix}_score_oof"].to_numpy(float)
    else:
        risk_column = "icvs_risk_score" if prefix == "icvs" else f"{prefix}_risk_score"
        if prefix == "radiomics":
            risk_column = "radiomics_score"
        risk = table[risk_column].to_numpy(float)
    result = pd.DataFrame(
        {
            patient_col: table[patient_col].astype(str),
            "model": model,
            "risk_score": risk,
        }
    )
    for horizon in horizons:
        result[f"pfs_{int(horizon)}m"] = table[f"{prefix}_pfs_{int(horizon)}m"].to_numpy(float)
    return result


def command_assemble_predictions(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config)
    specifications = [
        (args.clinical, "clinical", "clinical", False),
        (args.radiomics, "radiomics", "radiomics", False),
        (args.resnet, "3d_cnn", "resnet", True),
        (args.vit, "3d_vit", "vit", True),
        (args.icvs, "icvs", "icvs", False),
    ]
    tables = [
        _standard_prediction_table(
            frame,
            config,
            Path(path),
            model=model,
            prefix=prefix,
            deep=deep,
        )
        for path, model, prefix, deep in specifications
        if path is not None
    ]
    if not tables:
        raise ValueError("At least one model prediction file is required.")
    result = pd.concat(tables, ignore_index=True)
    if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise ValueError("Assembled predictions contain missing or nonfinite values.")
    _write_table(result, args.output)


def command_evaluate(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config)
    predictions = pd.read_csv(args.predictions)
    evaluate_models(frame, predictions, config, args.output_dir)


def command_explain(args: argparse.Namespace) -> None:
    frame, config = _load_manifest_and_config(args)
    validate_manifest(frame, config)
    scores = pd.read_csv(args.vit_scores)
    patient_col = config.column("patient_id")
    merged = frame.merge(
        scores[[patient_col, "vit_score_final", "vit_score_oof"]],
        on=patient_col,
        how="left",
        validate="one_to_one",
    )
    if merged["vit_score_final"].isna().any():
        raise ValueError("Final ViT scores are missing for one or more patients.")
    background = merged.loc[
        merged[config.column("cohort")].eq(config.cohort("training"))
    ].reset_index(drop=True)
    explained = select_explanation_patients(merged, config, total=args.patients)
    exact_time_dependent_shapley(
        args.artifact,
        background,
        explained,
        config,
        args.output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icvs-gbm-pfs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    _common_parser(validate)
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--paths", choices=("none", "raw", "processed"), default="none")
    validate.set_defaults(function=command_validate)

    preprocess = subparsers.add_parser("preprocess")
    _common_parser(preprocess)
    preprocess.add_argument("--manifest", required=True, type=Path)
    preprocess.add_argument("--output-root", required=True, type=Path)
    preprocess.add_argument("--output-manifest", required=True, type=Path)
    preprocess.set_defaults(function=command_preprocess)

    prepare_nnunet = subparsers.add_parser("prepare-nnunet")
    _common_parser(prepare_nnunet)
    prepare_nnunet.add_argument("--manifest", required=True, type=Path)
    prepare_nnunet.add_argument("--nnunet-raw", required=True, type=Path)
    prepare_nnunet.set_defaults(function=command_prepare_nnunet)

    train_nnunet = subparsers.add_parser("train-nnunet")
    _common_parser(train_nnunet)
    train_nnunet.add_argument("--nnunet-raw", required=True, type=Path)
    train_nnunet.add_argument("--nnunet-preprocessed", required=True, type=Path)
    train_nnunet.add_argument("--nnunet-results", required=True, type=Path)
    train_nnunet.add_argument("--trainer", default="nnUNetTrainer")
    train_nnunet.set_defaults(function=command_train_nnunet)

    predict_nnunet = subparsers.add_parser("predict-nnunet")
    _common_parser(predict_nnunet)
    predict_nnunet.add_argument("--input-dir", required=True, type=Path)
    predict_nnunet.add_argument("--output-dir", required=True, type=Path)
    predict_nnunet.add_argument("--nnunet-raw", required=True, type=Path)
    predict_nnunet.add_argument("--nnunet-preprocessed", required=True, type=Path)
    predict_nnunet.add_argument("--nnunet-results", required=True, type=Path)
    predict_nnunet.add_argument("--trainer", default="nnUNetTrainer")
    predict_nnunet.set_defaults(function=command_predict_nnunet)

    segmentation = subparsers.add_parser("evaluate-segmentation")
    _common_parser(segmentation)
    segmentation.add_argument("--manifest", required=True, type=Path)
    segmentation.add_argument("--reference-column", default="reference_mask_path")
    segmentation.add_argument("--prediction-column", default="prediction_mask_path")
    segmentation.add_argument("--bootstrap-resamples", type=int, default=1000)
    segmentation.add_argument("--output-dir", required=True, type=Path)
    segmentation.set_defaults(function=command_evaluate_segmentation)

    extract_radiomics = subparsers.add_parser("extract-radiomics")
    _common_parser(extract_radiomics)
    extract_radiomics.add_argument("--manifest", required=True, type=Path)
    extract_radiomics.add_argument("--parameters", required=True, type=Path)
    extract_radiomics.add_argument("--output", required=True, type=Path)
    extract_radiomics.set_defaults(function=command_extract_radiomics)

    train_radiomics = subparsers.add_parser("train-radiomics")
    _common_parser(train_radiomics)
    train_radiomics.add_argument("--manifest", required=True, type=Path)
    train_radiomics.add_argument("--features", required=True, type=Path)
    train_radiomics.add_argument("--output-dir", required=True, type=Path)
    train_radiomics.set_defaults(function=command_train_radiomics)

    clinical = subparsers.add_parser("fit-clinical")
    _common_parser(clinical)
    clinical.add_argument("--manifest", required=True, type=Path)
    clinical.add_argument("--output-dir", required=True, type=Path)
    clinical.set_defaults(function=command_fit_clinical)

    duration = subparsers.add_parser("select-duration")
    _common_parser(duration)
    duration.add_argument("--manifest", required=True, type=Path)
    duration.add_argument("--model", required=True, choices=("vit", "resnet"))
    duration.add_argument("--cache-dir", type=Path)
    duration.add_argument("--device")
    duration.add_argument("--output", required=True, type=Path)
    duration.set_defaults(function=command_select_duration)

    deep = subparsers.add_parser("train-deep")
    _common_parser(deep)
    deep.add_argument("--manifest", required=True, type=Path)
    deep.add_argument("--model", required=True, choices=("vit", "resnet"))
    deep.add_argument("--output-dir", required=True, type=Path)
    deep.add_argument("--cache-dir", type=Path)
    deep.add_argument("--device")
    deep.add_argument("--epochs", type=int)
    deep.set_defaults(function=command_train_deep)

    icvs = subparsers.add_parser("fit-icvs")
    _common_parser(icvs)
    icvs.add_argument("--manifest", required=True, type=Path)
    icvs.add_argument("--vit-scores", required=True, type=Path)
    icvs.add_argument("--output-dir", required=True, type=Path)
    icvs.set_defaults(function=command_fit_icvs)

    assemble = subparsers.add_parser("assemble-predictions")
    _common_parser(assemble)
    assemble.add_argument("--manifest", required=True, type=Path)
    assemble.add_argument("--clinical", type=Path)
    assemble.add_argument("--radiomics", type=Path)
    assemble.add_argument("--resnet", type=Path)
    assemble.add_argument("--vit", type=Path)
    assemble.add_argument("--icvs", type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    assemble.set_defaults(function=command_assemble_predictions)

    evaluate = subparsers.add_parser("evaluate")
    _common_parser(evaluate)
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.set_defaults(function=command_evaluate)

    explain = subparsers.add_parser("explain-icvs")
    _common_parser(explain)
    explain.add_argument("--manifest", required=True, type=Path)
    explain.add_argument("--vit-scores", required=True, type=Path)
    explain.add_argument("--artifact", required=True, type=Path)
    explain.add_argument("--patients", type=int, default=500)
    explain.add_argument("--output-dir", required=True, type=Path)
    explain.set_defaults(function=command_explain)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
