# ICVS-GBM-PFS: MRI phenotype modeling for progression-free survival in glioblastoma

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

This repository contains source code and fixed analysis parameters. It does not contain patient records, MRI volumes, masks, RNA-seq matrices, model checkpoints, or patient-level predictions.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[radiomics,segmentation,test]"
```

PyTorch must match the locally installed CUDA runtime. The study configuration records the framework-level parameters, while each cross-fitting run writes an `environment.json` file containing the installed package versions and compute device.

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
icvs-gbm-pfs validate-manifest \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_MANIFEST" \
  --paths raw
```

## MRI preprocessing

The preprocessing command rigidly registers T1-weighted, T2-weighted, and T2-FLAIR volumes to contrast-enhanced T1-weighted imaging; applies N4 bias-field correction; uses the supplied brain mask for skull exclusion; resamples all volumes to 1.0 x 1.0 x 5.0 mm; performs patient- and sequence-specific z-standardization; and builds the combined tumor-peritumoral VOI.

```bash
icvs-gbm-pfs preprocess \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_MANIFEST" \
  --output-root "$ICVS_GBM_PFS_PROCESSED_ROOT" \
  --output-manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST"
```

## Tumor-core segmentation

Only training-cohort masks are copied into the nnU-Net development directory. Planning, preprocessing, and all five folds use explicit storage locations. Validation cohorts are processed only by the locked five-fold ensemble.

```bash
icvs-gbm-pfs prepare-nnunet \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --nnunet-raw "$NNUNET_RAW"

icvs-gbm-pfs train-nnunet \
  --config configs/study.yaml \
  --nnunet-raw "$NNUNET_RAW" \
  --nnunet-preprocessed "$NNUNET_PREPROCESSED" \
  --nnunet-results "$NNUNET_RESULTS"

icvs-gbm-pfs prepare-nnunet-inference \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --input-dir "$ICVS_GBM_PFS_NNUNET_INPUT" \
  --prediction-dir "$ICVS_GBM_PFS_NNUNET_PREDICTIONS" \
  --output-manifest "$ICVS_GBM_PFS_SEGMENTATION_MANIFEST"

icvs-gbm-pfs predict-nnunet \
  --config configs/study.yaml \
  --input-dir "$ICVS_GBM_PFS_NNUNET_INPUT" \
  --output-dir "$ICVS_GBM_PFS_NNUNET_PREDICTIONS" \
  --nnunet-raw "$NNUNET_RAW" \
  --nnunet-preprocessed "$NNUNET_PREPROCESSED" \
  --nnunet-results "$NNUNET_RESULTS"
```

Segmentation assessment operates on unedited automatic masks and reports Dice, surface Dice at 2 mm, sensitivity, HD95, relative volume error, and patient-level volume pairs for Bland-Altman analysis.
Cases without predicted foreground retain infinite patient-level HD95 values. Cohort HD95 summaries report the evaluable and nonfinite case counts separately and calculate intervals only from finite distances.

```bash
icvs-gbm-pfs evaluate-segmentation \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_SEGMENTATION_MANIFEST" \
  --output-dir "$ICVS_GBM_PFS_SEGMENTATION_RESULTS"
```

## Radiomics

`configs/radiomics.yaml` fixes radiomics-only 1.0-mm isotropic resampling, the discretization width, feature classes, coif1 wavelet decomposition, and Laplacian-of-Gaussian scales. Extraction uses the combined VOI and all four registered sequences. Modality-independent shape features are retained once, while first-order and texture features remain sequence-specific. Standardization is fitted in the training cohort and applied unchanged to locked cohorts. Within each cross-validation split, univariable screening and standardization are refitted using only that split's fitting patients. Fold-specific LASSO-Cox paths are compared by relative penalty position, the penalty is selected with the one-standard-error rule, and the final screened multivariable Cox model is fitted once in the complete training cohort.

```bash
icvs-gbm-pfs extract-radiomics \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --parameters configs/radiomics.yaml \
  --output "$ICVS_GBM_PFS_RADIOMICS_FEATURES"

icvs-gbm-pfs train-radiomics \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --features "$ICVS_GBM_PFS_RADIOMICS_FEATURES" \
  --output-dir "$ICVS_GBM_PFS_RADIOMICS_RESULTS"
```

## Deep survival models

The 3D-ViT uses 16 x 16 x 4-voxel patches, a 256-dimensional embedding, six pre-normalized transformer blocks, eight attention heads, and a continuous Cox log-risk output. The convolutional comparator is an anisotropic four-channel 3D ResNet-18 with the same survival endpoint and head width.

The duration-selection command uses only an 80:20 patient-level partition of the training cohort. Final execution uses the fixed duration, performs five-fold cross-fitting for training-cohort scores, refits once on the full training cohort, and then scores the temporal and spatial cohorts without model selection or refitting.

```bash
icvs-gbm-pfs select-duration \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --model vit \
  --cache-dir "$ICVS_GBM_PFS_VOLUME_CACHE" \
  --output "$ICVS_GBM_PFS_VIT_DURATION"

icvs-gbm-pfs select-duration \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --model resnet \
  --cache-dir "$ICVS_GBM_PFS_VOLUME_CACHE" \
  --output "$ICVS_GBM_PFS_RESNET_DURATION"

icvs-gbm-pfs train-deep \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --model vit \
  --cache-dir "$ICVS_GBM_PFS_VOLUME_CACHE" \
  --duration-file "$ICVS_GBM_PFS_VIT_DURATION" \
  --output-dir "$ICVS_GBM_PFS_VIT_RESULTS"

icvs-gbm-pfs train-deep \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --model resnet \
  --cache-dir "$ICVS_GBM_PFS_VOLUME_CACHE" \
  --duration-file "$ICVS_GBM_PFS_RESNET_DURATION" \
  --output-dir "$ICVS_GBM_PFS_RESNET_RESULTS"
```

The augmentation path applies one spatial transform to all four MRI channels and the VOI mask. It includes bounded rotation, translation, scaling, intensity shift and scale, Gaussian noise, and smoothing while preserving zero-valued background outside the transformed VOI. Left-right flipping is not used. Training-cohort survival probabilities are estimated with the baseline hazard from the corresponding cross-fitting training fold; the held-out patient's outcome is not used for that prediction.

## Clinical and integrated models

The clinical comparator uses the three retained predictors with a Breslow-tied Cox model. ICVS is fitted once using standardized out-of-fold ViT scores and training outcomes. Training performance and probabilities use out-of-bag trees; validation uses the locked full forest and the final-model ViT score transformation estimated in the training cohort.

```bash
icvs-gbm-pfs fit-clinical \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --output-dir "$ICVS_GBM_PFS_CLINICAL_RESULTS"

icvs-gbm-pfs fit-icvs \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --vit-scores "$ICVS_GBM_PFS_VIT_RESULTS/vit_scores.csv" \
  --output-dir "$ICVS_GBM_PFS_ICVS_RESULTS"
```

## Evaluation and interpretation

The prediction assembly command converts model-specific outputs to one patient-model table. Training deep-model rows use out-of-fold scores and matching survival estimates. ICVS training rows use out-of-bag outputs. Temporal and spatial validation rows use locked-model outputs.

```bash
icvs-gbm-pfs assemble-predictions \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --clinical "$ICVS_GBM_PFS_CLINICAL_RESULTS/clinical_predictions.csv" \
  --radiomics "$ICVS_GBM_PFS_RADIOMICS_RESULTS/radiomics_scores.csv" \
  --resnet "$ICVS_GBM_PFS_RESNET_RESULTS/resnet_scores.csv" \
  --vit "$ICVS_GBM_PFS_VIT_RESULTS/vit_scores.csv" \
  --icvs "$ICVS_GBM_PFS_ICVS_RESULTS/icvs_predictions.csv" \
  --output "$ICVS_GBM_PFS_LOCKED_PREDICTIONS"

icvs-gbm-pfs evaluate \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --predictions "$ICVS_GBM_PFS_LOCKED_PREDICTIONS" \
  --output-dir "$ICVS_GBM_PFS_EVALUATION_RESULTS"
```

Evaluation includes Harrell concordance, dynamic AUC from 6 to 36 months, IPCW Brier scores, integrated Brier score, grouped 12-month calibration, locked-median Kaplan-Meier stratification, complete unadjusted and adjusted Cox tables, rank-based Schoenfeld residual tests of proportional hazards, and paired patient-level bootstrap comparisons. Benjamini-Hochberg adjustments are reported for model-wise risk-stratification tests within each cohort and pairwise model comparisons within each cohort and metric. The biological subset is labeled as nested and is never reported as independent validation. The workflow evaluates prognostic performance; it does not claim that a clinical decision threshold or net benefit has been established.

Time-dependent ICVS interpretation evaluates every coalition of the four predictors against the complete training-cohort background and verifies additivity for every patient and horizon.

```bash
icvs-gbm-pfs explain-icvs \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --vit-scores "$ICVS_GBM_PFS_VIT_RESULTS/vit_scores.csv" \
  --artifact "$ICVS_GBM_PFS_ICVS_RESULTS/icvs_model.joblib" \
  --patients 500 \
  --output-dir "$ICVS_GBM_PFS_EXPLANATION_RESULTS"
```

## Transcriptomic analysis

The biological cohort is generated directly from the prespecified nested subset. It uses out-of-fold ViT scores and carries the median cutoff estimated from the complete training cohort, preventing final-model scores or a subset-derived threshold from entering this analysis.

```bash
icvs-gbm-pfs prepare-biological-cohort \
  --config configs/study.yaml \
  --manifest "$ICVS_GBM_PFS_PROCESSED_MANIFEST" \
  --vit-scores "$ICVS_GBM_PFS_VIT_RESULTS/vit_scores.csv" \
  --output "$ICVS_GBM_PFS_BIOLOGICAL_COHORT"
```

The R workflow restricts counts to GENCODE protein-coding genes, applies the prespecified expression filter, fits DESeq2, runs Hallmark preranked GSEA and ssGSEA, computes groupwise and continuous pathway statistics with separate FDR families, and evaluates the prespecified total-macrophage aggregate separately from the 22 exploratory LM22 populations.

```bash
Rscript R/biological_analysis.R \
  --cohort "$ICVS_GBM_PFS_BIOLOGICAL_COHORT" \
  --counts "$ICVS_GBM_PFS_RNA_COUNTS" \
  --gene-annotation "$ICVS_GBM_PFS_GENCODE_REFERENCE" \
  --hallmark-gmt "$ICVS_GBM_PFS_HALLMARK_GMT" \
  --lm22-fractions "$ICVS_GBM_PFS_LM22_FRACTIONS" \
  --bootstrap-resamples 3000 \
  --seed 2026 \
  --output-dir "$ICVS_GBM_PFS_BIOLOGY_RESULTS"
```

## Quality controls

```bash
ruff check src tests
pytest
```

The automated checks cover manifest isolation, Cox loss and baseline-hazard calculations, model output geometry, VOI construction, and segmentation metrics. Full data-dependent verification must additionally reconcile patient counts, event counts, split assignments, model hashes, preprocessing logs, and every reported table and figure against the governed analysis release.

## License

The source code is released under the MIT License. Data access, trained weights, and institutional model artifacts are not granted by this software license.

## Citation

Citation metadata for this release is provided in `CITATION.cff`. When the associated article has a persistent identifier, add that article citation alongside the software citation.
