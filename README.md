# DLR3: MRI phenotype modeling for progression-free survival in glioblastoma

This repository contains the analysis code for patient-level MRI preprocessing, tumor-core segmentation, radiomics, three-dimensional deep survival modeling, integrated clinical-imaging survival modeling, locked-cohort evaluation, time-dependent model interpretation, and transcriptomic analysis.

The implementation follows the accompanying study protocol:

- Four registered MRI sequences: T1-weighted, T2-weighted, T2-FLAIR, and contrast-enhanced T1-weighted imaging.
- A corrected tumor core combined with a 10-mm in-plane peritumoral margin.
- A 3D vision transformer and a 3D ResNet-18 comparator trained with a Cox survival objective.
- Patient-level five-fold cross-fitting within the training cohort.
- Temporal and spatial validation with locked models, preprocessing parameters, and risk cutoffs.
- An Integrated Clinical-ViT Survival model using age, MGMT promoter methylation, extent of resection, and the continuous ViT score.
- Differential expression, Hallmark GSEA, patient-level ssGSEA, and LM22 immune-cell analyses.

## Repository scope

This repository contains source code and fixed analysis parameters. It does not contain patient records, MRI volumes, masks, RNA-seq matrices, model checkpoints, or patient-level predictions. Access to institutional data remains subject to ethics approval, data-use agreements, and local governance. The reported numerical results cannot be recreated without the governed study data and locked model artifacts.

All input identifiers must be nonidentifying surrogates. Names, medical-record numbers, accession numbers, dates, and raw DICOM metadata must not be placed in the manifest or committed to version control.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[radiomics,segmentation,test]"
```

PyTorch must match the locally installed CUDA runtime. The study configuration records the framework-level parameters, while each run writes an `environment.json` file containing the installed package versions and compute device.

The biological analysis requires R and the following packages:

- Bioconductor: `DESeq2`, `fgsea`, and `GSVA`
- CRAN: `dplyr`, `readr`, and `tibble`

## Patient manifest

The pipeline uses one row per patient. Relative image paths are resolved against the manifest location. The formal field definition is provided in `configs/manifest.schema.json`.

Required fields before MRI preprocessing:

| Field | Definition |
|---|---|
| `patient_id` | Unique nonidentifying patient surrogate |
| `cohort` | `training`, `temporal_validation`, or `spatial_validation` |
| `center_id` | Nonidentifying center label |
| `pfs_months` | Observed progression-free survival time in months |
| `pfs_event` | `1` for progression or death, `0` for censoring |
| `biological_subset` | `1` only for the nested training-cohort transcriptomic subset |
| `age_years` | Age at the study-defined baseline |
| `mgmt_methylated` | `1` for methylated, `0` for unmethylated |
| `non_gross_total_resection` | `1` for non-gross-total resection, `0` for gross-total resection |
| `t1_path` | T1-weighted NIfTI volume |
| `t2_path` | T2-weighted NIfTI volume |
| `flair_path` | T2-FLAIR NIfTI volume |
| `ce_t1_path` | Contrast-enhanced T1-weighted NIfTI volume |
| `brain_mask_path` | Brain mask in the contrast-enhanced T1 reference space |
| `tumor_mask_path` | Reader-approved tumor-core reference mask |

The validation command blocks duplicate patients, invalid survival endpoints, path failures, a biological subset outside the training cohort, and overlap between development and spatial-validation centers.

```bash
dlr3 validate-manifest \
  --config configs/study.yaml \
  --manifest "$DLR3_MANIFEST" \
  --paths raw
```

## MRI preprocessing

The preprocessing command rigidly registers T1-weighted, T2-weighted, and T2-FLAIR volumes to contrast-enhanced T1-weighted imaging; applies N4 bias-field correction; uses the supplied brain mask for skull exclusion; resamples all volumes to 1.0 x 1.0 x 5.0 mm; performs patient- and sequence-specific z-standardization; and builds the combined tumor-peritumoral VOI.

```bash
dlr3 preprocess \
  --config configs/study.yaml \
  --manifest "$DLR3_MANIFEST" \
  --output-root "$DLR3_PROCESSED_ROOT" \
  --output-manifest "$DLR3_PROCESSED_MANIFEST"
```

## Tumor-core segmentation

Only training-cohort masks are copied into the nnU-Net development directory. Planning, preprocessing, and all five folds use explicit storage locations. Validation cohorts are processed only by the locked five-fold ensemble.

```bash
dlr3 prepare-nnunet \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --nnunet-raw "$NNUNET_RAW"

dlr3 train-nnunet \
  --config configs/study.yaml \
  --nnunet-raw "$NNUNET_RAW" \
  --nnunet-preprocessed "$NNUNET_PREPROCESSED" \
  --nnunet-results "$NNUNET_RESULTS"
```

Segmentation assessment operates on unedited automatic masks and reports Dice, surface Dice at 2 mm, sensitivity, HD95, relative volume error, and patient-level volume pairs for Bland-Altman analysis.

```bash
dlr3 evaluate-segmentation \
  --config configs/study.yaml \
  --manifest "$DLR3_SEGMENTATION_MANIFEST" \
  --output-dir "$DLR3_SEGMENTATION_RESULTS"
```

## Radiomics

`configs/radiomics.yaml` fixes the discretization width, feature classes, coif1 wavelet decomposition, and Laplacian-of-Gaussian scales. Extraction uses the combined VOI and all four registered sequences. Standardization is fitted in the training cohort and applied unchanged to locked cohorts. The model performs univariable Cox screening, 10-fold LASSO-Cox penalty selection with the one-standard-error rule, and final multivariable Cox fitting.

```bash
dlr3 extract-radiomics \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --parameters configs/radiomics.yaml \
  --output "$DLR3_RADIOMICS_FEATURES"

dlr3 train-radiomics \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --features "$DLR3_RADIOMICS_FEATURES" \
  --output-dir "$DLR3_RADIOMICS_RESULTS"
```

## Deep survival models

The 3D-ViT uses 16 x 16 x 4-voxel patches, a 256-dimensional embedding, six pre-normalized transformer blocks, eight attention heads, and a continuous Cox log-risk output. The convolutional comparator is an anisotropic four-channel 3D ResNet-18 with the same survival endpoint and head width.

The duration-selection command uses only an 80:20 patient-level partition of the training cohort. Final execution uses the fixed duration, performs five-fold cross-fitting for training-cohort scores, refits once on the full training cohort, and then scores the temporal and spatial cohorts without model selection or refitting.

```bash
dlr3 select-duration \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --model vit \
  --cache-dir "$DLR3_VOLUME_CACHE" \
  --output "$DLR3_VIT_DURATION"

dlr3 train-deep \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --model vit \
  --cache-dir "$DLR3_VOLUME_CACHE" \
  --output-dir "$DLR3_VIT_RESULTS"

dlr3 train-deep \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --model resnet \
  --cache-dir "$DLR3_VOLUME_CACHE" \
  --output-dir "$DLR3_RESNET_RESULTS"
```

The augmentation path applies one spatial transform to all four MRI channels. It includes bounded rotation, translation, scaling, intensity shift and scale, Gaussian noise, and smoothing. Left-right flipping is not used.

## Clinical and integrated models

The clinical comparator uses the three retained predictors with a Breslow-tied Cox model. ICVS is fitted once using standardized out-of-fold ViT scores and training outcomes. Training performance and probabilities use out-of-bag trees; validation uses the locked full forest and the final-model ViT score transformation estimated in the training cohort.

```bash
dlr3 fit-clinical \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --output-dir "$DLR3_CLINICAL_RESULTS"

dlr3 fit-icvs \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --vit-scores "$DLR3_VIT_RESULTS/vit_scores.csv" \
  --output-dir "$DLR3_ICVS_RESULTS"
```

## Evaluation and interpretation

The prediction assembly command converts model-specific outputs to one patient-model table. Training deep-model rows use out-of-fold scores and matching survival estimates. ICVS training rows use out-of-bag outputs. Temporal and spatial validation rows use locked-model outputs.

```bash
dlr3 assemble-predictions \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --clinical "$DLR3_CLINICAL_RESULTS/clinical_predictions.csv" \
  --radiomics "$DLR3_RADIOMICS_RESULTS/radiomics_scores.csv" \
  --resnet "$DLR3_RESNET_RESULTS/resnet_scores.csv" \
  --vit "$DLR3_VIT_RESULTS/vit_scores.csv" \
  --icvs "$DLR3_ICVS_RESULTS/icvs_predictions.csv" \
  --output "$DLR3_LOCKED_PREDICTIONS"

dlr3 evaluate \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --predictions "$DLR3_LOCKED_PREDICTIONS" \
  --output-dir "$DLR3_EVALUATION_RESULTS"
```

Evaluation includes Harrell concordance, dynamic AUC from 6 to 36 months, IPCW Brier scores, integrated Brier score, grouped 12-month calibration, locked-median Kaplan-Meier stratification, adjusted Cox regression, and paired patient-level bootstrap comparisons. The biological subset is labeled as nested and is never reported as independent validation.

Time-dependent ICVS interpretation evaluates every coalition of the four predictors against the complete training-cohort background and verifies additivity for every patient and horizon.

```bash
dlr3 explain-icvs \
  --config configs/study.yaml \
  --manifest "$DLR3_PROCESSED_MANIFEST" \
  --vit-scores "$DLR3_VIT_RESULTS/vit_scores.csv" \
  --artifact "$DLR3_ICVS_RESULTS/icvs_model.joblib" \
  --patients 500 \
  --output-dir "$DLR3_EXPLANATION_RESULTS"
```

## Transcriptomic analysis

The R workflow restricts counts to GENCODE protein-coding genes, applies the prespecified expression filter, fits DESeq2, runs Hallmark preranked GSEA and ssGSEA, computes groupwise and continuous pathway statistics with separate FDR families, and evaluates the prespecified total-macrophage aggregate separately from the 22 exploratory LM22 populations.

```bash
Rscript R/biological_analysis.R \
  --cohort "$DLR3_BIOLOGICAL_COHORT" \
  --counts "$DLR3_RNA_COUNTS" \
  --gene-annotation "$DLR3_GENCODE_REFERENCE" \
  --hallmark-gmt "$DLR3_HALLMARK_GMT" \
  --lm22-fractions "$DLR3_LM22_FRACTIONS" \
  --vit-cutoff "$DLR3_VIT_CUTOFF" \
  --bootstrap-resamples 3000 \
  --seed 2026 \
  --output-dir "$DLR3_BIOLOGY_RESULTS"
```

## Quality controls

```bash
ruff check src tests
pytest
```

The automated checks cover manifest isolation, Cox loss and baseline-hazard calculations, model output geometry, VOI construction, and segmentation metrics. Full data-dependent verification must additionally reconcile patient counts, event counts, split assignments, model hashes, preprocessing logs, and every reported table and figure against the governed analysis release.

## License

The source code is released under the MIT License. Data access, trained weights, and institutional model artifacts are not granted by this software license.
