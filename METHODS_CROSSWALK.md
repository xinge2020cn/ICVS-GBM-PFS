# Methods-to-code crosswalk

This crosswalk identifies the public implementation for each result-bearing analysis component. It does not claim that protected patient data, MRI volumes, RNA-seq matrices, trained weights, or fitted model artifacts are publicly available.

| Analysis component | Fixed parameters | Public implementation | Primary outputs | Availability boundary |
|---|---|---|---|---|
| Cohort and endpoint controls | `configs/study.yaml`, `configs/manifest.schema.json` | `validate-manifest`, `summarize-cohorts` | Validation audit, Table 1 source rows, cross-cohort tests, reverse Kaplan-Meier follow-up | Patient records and institutional identifiers are excluded. |
| MRI preprocessing | `configs/study.yaml` | `icvs_gbm_pfs.preprocessing.preprocess_manifest` | Registered volumes, masks, combined VOI, processed manifest | MRI volumes and masks are excluded. |
| Tumor-core segmentation development | `configs/study.yaml`, nnU-Net 2.4.2 | `prepare-nnunet`, `train-nnunet` | nnU-Net folds and validation predictions | Epoch, iteration, optimizer, oversampling, deep-supervision, and mixed-precision settings are recorded; fitted weights are excluded. |
| Training-cohort segmentation assessment | Five patient-level folds | `collect-nnunet-oof`, `assemble-segmentation-manifest`, `evaluate-segmentation` | Patient-level and cohort-level segmentation metrics, volume bias and limits of agreement | Requires governed fold outputs and reference masks. |
| Locked validation segmentation | Five-fold ensemble | `prepare-nnunet-inference`, `predict-nnunet`, `evaluate-segmentation` | Temporal and spatial validation metrics | Requires governed weights and validation images. |
| Radiomic feature extraction | `configs/radiomics.yaml` | `extract-radiomics` | IBSI-aligned patient-feature table | Derived patient-level features are excluded. |
| Radiomics model | Univariable Cox screening, ten-fold LASSO-Cox, one-standard-error rule | `train-radiomics` | Screening, penalty selection, selected features, predictions, fitted artifact | Screening and standardization are estimated in the complete training cohort and applied unchanged to locked cohorts, following the reported analysis sequence. |
| Clinical predictor selection | Univariable P < .20; multivariable P < .05; penalizer 0.001 | `fit-clinical` | Complete Table 2-compatible selection table, predictions, fitted artifact | The command verifies that age, extent of resection, and MGMT are retained before fitting the locked comparator. |
| 3D-CNN and 3D-ViT duration selection | `configs/study.yaml` | `select-duration` | Development/tuning partition and selected epoch | Requires processed volumes. |
| 3D-CNN and 3D-ViT modeling | Five-fold cross-fitting and full-training refit | `train-deep` | Out-of-fold training scores, locked validation scores, probabilities, checkpoints, environment report | Checkpoints and patient-level scores are excluded. |
| Integrated Clinical-ViT Survival model | 500-tree random survival forest and six fixed horizons | `fit-icvs` | Out-of-bag training predictions, locked validation predictions, fitted artifact | The fitted artifact is excluded. |
| Performance and risk-stratification analysis | `configs/study.yaml` | `assemble-predictions`, `evaluate` | Concordance, dynamic AUC, ROC coordinates, Brier scores, calibration, Kaplan-Meier curve data, Cox and comparison tables | Requires governed patient-level predictions. |
| Time-dependent ICVS interpretation | Four predictors and complete training background | `explain-icvs` | Patient-level Shapley values, mean absolute contributions, patient-bootstrap confidence intervals, 12-month ViT dependence curve | Requires the fitted artifact and governed score table. |
| Transcriptomic analysis | Fixed seed and declared analysis arguments | `prepare-biological-cohort`, `R/biological_analysis.R` | Differential expression, GSEA, ssGSEA, leading-edge heatmap data, ranked evidence, immune-cell and pathway tables | RNA-seq matrices, LM22 fractions, and patient-level annotations are excluded. Upstream read processing and licensed signature execution remain governed inputs. |
| Research web interface | Locked artifact and standardized 3D-ViT score | `deployment/app.py`, `deployment/predictor.py` | Browser interface, API, metadata and health endpoints | The service will not start without an explicitly supplied fitted artifact. |

## Reproducibility boundary

The repository is executable source code, not a public clinical dataset or a trained-model release. Exact numerical reproduction requires the governed analysis inputs, split assignments, fitted artifacts, and preprocessing logs. The software environment explicitly reported in the Supplementary Methods is recorded in `environment/study-reported.yaml`; the maintained package is tested separately on the Python versions listed in continuous integration.

## Archival release

For a citable software release, create a signed repository tag and archive that release in a DOI-issuing repository. Do not add an archival DOI to citation metadata until the DOI has been issued. If trained weights or derived feature tables are released later, publish their license, provenance, intended-use boundary, and persistent identifier separately.
