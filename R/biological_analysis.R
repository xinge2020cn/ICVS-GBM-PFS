suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
  library(fgsea)
  library(GSVA)
  library(readr)
  library(tibble)
})

options(stringsAsFactors = FALSE)

parse_arguments <- function(values) {
  if (length(values) %% 2 != 0) {
    stop("Arguments must be provided as --name value pairs.")
  }
  result <- list()
  for (index in seq(1, length(values), by = 2)) {
    key <- sub("^--", "", values[[index]])
    result[[key]] <- values[[index + 1]]
  }
  result
}

require_argument <- function(arguments, name) {
  value <- arguments[[name]]
  if (is.null(value) || !nzchar(value)) {
    stop(sprintf("Missing required argument --%s.", name))
  }
  value
}

read_gmt <- function(path) {
  lines <- readLines(path, warn = FALSE)
  parsed <- strsplit(lines, "\t", fixed = TRUE)
  sets <- lapply(parsed, function(value) unique(value[-c(1, 2)]))
  names(sets) <- vapply(parsed, `[[`, character(1), 1)
  sets
}

rank_biserial <- function(high, low) {
  high <- high[is.finite(high)]
  low <- low[is.finite(low)]
  statistic <- unname(wilcox.test(high, low, exact = FALSE)$statistic)
  2 * statistic / (length(high) * length(low)) - 1
}

bootstrap_interval <- function(data_size, statistic, resamples, seed) {
  set.seed(seed)
  values <- replicate(resamples, statistic(sample.int(data_size, data_size, replace = TRUE)))
  unname(quantile(values, c(0.025, 0.975), na.rm = TRUE))
}

pathway_statistics <- function(scores, cohort, resamples, seed) {
  rows <- lapply(seq_len(nrow(scores)), function(index) {
    values <- as.numeric(scores[index, ])
    high <- values[cohort$vit_group == "High"]
    low <- values[cohort$vit_group == "Low"]
    group_test <- wilcox.test(high, low, exact = FALSE)
    effect <- rank_biserial(high, low)
    effect_ci <- bootstrap_interval(
      length(values),
      function(selected) {
        selected_values <- values[selected]
        selected_group <- cohort$vit_group[selected]
        if (length(unique(selected_group)) < 2) return(NA_real_)
        rank_biserial(
          selected_values[selected_group == "High"],
          selected_values[selected_group == "Low"]
        )
      },
      resamples,
      seed + index
    )
    correlation <- suppressWarnings(cor.test(values, cohort$vit_score, method = "spearman", exact = FALSE))
    correlation_ci <- bootstrap_interval(
      length(values),
      function(selected) suppressWarnings(
        cor(values[selected], cohort$vit_score[selected], method = "spearman")
      ),
      resamples,
      seed + 1000L + index
    )
    tibble(
      pathway = rownames(scores)[index],
      high_median = median(high),
      low_median = median(low),
      rank_biserial = effect,
      rank_biserial_ci_low = effect_ci[[1]],
      rank_biserial_ci_high = effect_ci[[2]],
      group_p_value = group_test$p.value,
      spearman_rho = unname(correlation$estimate),
      spearman_ci_low = correlation_ci[[1]],
      spearman_ci_high = correlation_ci[[2]],
      correlation_p_value = correlation$p.value
    )
  })
  bind_rows(rows) %>%
    mutate(
      group_fdr = p.adjust(group_p_value, method = "BH"),
      correlation_fdr = p.adjust(correlation_p_value, method = "BH")
    )
}

cell_statistics <- function(fractions, cohort, resamples, seed) {
  cell_columns <- setdiff(names(fractions), "patient_id")
  rows <- lapply(seq_along(cell_columns), function(index) {
    cell_type <- cell_columns[[index]]
    values <- fractions[[cell_type]]
    high <- values[cohort$vit_group == "High"]
    low <- values[cohort$vit_group == "Low"]
    effect_ci <- bootstrap_interval(
      length(values),
      function(selected) {
        selected_values <- values[selected]
        selected_group <- cohort$vit_group[selected]
        if (length(unique(selected_group)) < 2) return(NA_real_)
        rank_biserial(
          selected_values[selected_group == "High"],
          selected_values[selected_group == "Low"]
        )
      },
      resamples,
      seed + index
    )
    correlation_ci <- bootstrap_interval(
      length(values),
      function(selected) suppressWarnings(
        cor(values[selected], cohort$vit_score[selected], method = "spearman")
      ),
      resamples,
      seed + 1000L + index
    )
    group_test <- wilcox.test(high, low, exact = FALSE)
    correlation <- suppressWarnings(cor.test(values, cohort$vit_score, method = "spearman", exact = FALSE))
    tibble(
      cell_type = cell_type,
      high_median = median(high),
      low_median = median(low),
      rank_biserial = rank_biserial(high, low),
      rank_biserial_ci_low = effect_ci[[1]],
      rank_biserial_ci_high = effect_ci[[2]],
      group_p_value = group_test$p.value,
      spearman_rho = unname(correlation$estimate),
      spearman_ci_low = correlation_ci[[1]],
      spearman_ci_high = correlation_ci[[2]],
      correlation_p_value = correlation$p.value
    )
  })
  bind_rows(rows) %>%
    mutate(
      group_fdr = p.adjust(group_p_value, method = "BH"),
      correlation_fdr = p.adjust(correlation_p_value, method = "BH")
    )
}

arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
cohort_path <- require_argument(arguments, "cohort")
counts_path <- require_argument(arguments, "counts")
annotation_path <- require_argument(arguments, "gene-annotation")
hallmark_path <- require_argument(arguments, "hallmark-gmt")
lm22_path <- require_argument(arguments, "lm22-fractions")
output_dir <- require_argument(arguments, "output-dir")
vit_cutoff <- as.numeric(require_argument(arguments, "vit-cutoff"))
resamples <- as.integer(ifelse(is.null(arguments[["bootstrap-resamples"]]), 3000, arguments[["bootstrap-resamples"]]))
seed <- as.integer(ifelse(is.null(arguments[["seed"]]), 2026, arguments[["seed"]]))

if (!is.finite(vit_cutoff)) stop("The ViT cutoff must be finite.")
if (resamples < 100) stop("At least 100 bootstrap resamples are required.")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

cohort <- read_csv(cohort_path, show_col_types = FALSE) %>%
  select(patient_id, vit_score) %>%
  mutate(
    patient_id = as.character(patient_id),
    vit_group = factor(if_else(vit_score >= vit_cutoff, "High", "Low"), levels = c("Low", "High"))
  ) %>%
  arrange(patient_id)

if (anyDuplicated(cohort$patient_id)) stop("Cohort patient identifiers must be unique.")
if (any(!is.finite(cohort$vit_score))) stop("ViT scores must be finite.")
if (length(unique(cohort$vit_group)) != 2) stop("The locked cutoff must create two groups.")

annotation <- read_tsv(annotation_path, show_col_types = FALSE) %>%
  filter(gene_type == "protein_coding") %>%
  distinct(gene_symbol)
counts_frame <- read_csv(counts_path, show_col_types = FALSE)
gene_symbols <- as.character(counts_frame[[1]])
counts <- as.matrix(counts_frame[, -1])
storage.mode(counts) <- "integer"
rownames(counts) <- gene_symbols
if (anyDuplicated(rownames(counts))) stop("RNA count gene symbols must be unique.")
if (!all(cohort$patient_id %in% colnames(counts))) stop("RNA counts are missing cohort patients.")
counts <- counts[, cohort$patient_id, drop = FALSE]
counts <- counts[rownames(counts) %in% annotation$gene_symbol, , drop = FALSE]
minimum_samples <- ceiling(0.20 * ncol(counts))
keep <- rowSums(counts >= 10L) >= minimum_samples
counts <- counts[keep, , drop = FALSE]
if (nrow(counts) == 0) stop("No protein-coding genes passed the expression filter.")

write_csv(
  tibble(
    samples = ncol(counts),
    minimum_count = 10L,
    minimum_samples = minimum_samples,
    retained_genes = nrow(counts)
  ),
  file.path(output_dir, "rna_filtering_summary.csv")
)

sample_data <- data.frame(vit_group = cohort$vit_group, row.names = cohort$patient_id)
dds <- DESeqDataSetFromMatrix(countData = counts, colData = sample_data, design = ~ vit_group)
dds <- DESeq(dds, quiet = TRUE)
differential <- results(
  dds,
  contrast = c("vit_group", "High", "Low"),
  alpha = 0.05,
  independentFiltering = TRUE
) %>%
  as.data.frame() %>%
  rownames_to_column("gene_symbol") %>%
  rename(
    base_mean = baseMean,
    log2_fold_change = log2FoldChange,
    standard_error = lfcSE,
    wald_statistic = stat,
    p_value = pvalue,
    fdr = padj
  ) %>%
  mutate(
    differential_expression = case_when(
      fdr < 0.05 & log2_fold_change > 0.50 ~ "higher_in_high_vit",
      fdr < 0.05 & log2_fold_change < -0.50 ~ "lower_in_high_vit",
      TRUE ~ "not_significant"
    )
  ) %>%
  arrange(fdr, p_value)
write_csv(differential, file.path(output_dir, "differential_expression.csv"))

variance_stabilized <- assay(vst(dds, blind = FALSE))
gene_sets <- read_gmt(hallmark_path)
ranks <- differential$wald_statistic
names(ranks) <- differential$gene_symbol
ranks <- sort(ranks[is.finite(ranks)], decreasing = TRUE)
enrichment <- fgseaMultilevel(
  pathways = gene_sets,
  stats = ranks,
  minSize = 15,
  maxSize = 500,
  eps = 0,
  nPermSimple = 10000
) %>%
  as_tibble() %>%
  select(pathway, size, enrichmentScore, NES, pval, padj, leadingEdge) %>%
  rename(
    enrichment_score = enrichmentScore,
    normalized_enrichment_score = NES,
    p_value = pval,
    fdr = padj,
    leading_edge = leadingEdge
  ) %>%
  mutate(leading_edge = vapply(leading_edge, paste, collapse = ";", character(1))) %>%
  arrange(fdr, desc(abs(normalized_enrichment_score)))
write_csv(enrichment, file.path(output_dir, "hallmark_gsea.csv"))

if (exists("ssgseaParam", where = asNamespace("GSVA"), mode = "function")) {
  parameters <- GSVA::ssgseaParam(variance_stabilized, gene_sets, normalize = TRUE)
  ssgsea_scores <- GSVA::gsva(parameters, verbose = FALSE)
} else {
  ssgsea_scores <- GSVA::gsva(
    variance_stabilized,
    gene_sets,
    method = "ssgsea",
    kcdf = "Gaussian",
    abs.ranking = TRUE,
    verbose = FALSE
  )
}
ssgsea_scores <- ssgsea_scores[, cohort$patient_id, drop = FALSE]
pathway_results <- pathway_statistics(ssgsea_scores, cohort, resamples, seed) %>%
  left_join(
    enrichment %>% select(pathway, normalized_enrichment_score, gsea_fdr = fdr),
    by = "pathway"
  ) %>%
  mutate(
    directionally_concordant =
      sign(normalized_enrichment_score) == sign(rank_biserial) &
      sign(normalized_enrichment_score) == sign(spearman_rho),
    cross_analysis_consistent =
      directionally_concordant & gsea_fdr < 0.05 & group_fdr < 0.05 & correlation_fdr < 0.05
  )
write_csv(pathway_results, file.path(output_dir, "hallmark_patient_level_statistics.csv"))
write_csv(
  as.data.frame(ssgsea_scores) %>% rownames_to_column("pathway"),
  file.path(output_dir, "hallmark_ssgsea_scores.csv")
)

fractions <- read_csv(lm22_path, show_col_types = FALSE) %>%
  mutate(patient_id = as.character(patient_id))
if (anyDuplicated(fractions$patient_id)) stop("LM22 patient identifiers must be unique.")
if (!all(cohort$patient_id %in% fractions$patient_id)) stop("LM22 fractions are missing cohort patients.")
fractions <- fractions[match(cohort$patient_id, fractions$patient_id), ]
cell_columns <- setdiff(names(fractions), "patient_id")
fraction_matrix <- as.matrix(fractions[, cell_columns])
storage.mode(fraction_matrix) <- "double"
fraction_matrix[fraction_matrix < 0] <- 0
row_totals <- rowSums(fraction_matrix)
if (any(row_totals <= 0)) stop("LM22 fractions must have a positive row sum for every patient.")
fraction_matrix <- fraction_matrix / row_totals
fractions[, cell_columns] <- fraction_matrix
macrophage_columns <- c("Macrophages M0", "Macrophages M1", "Macrophages M2")
if (!all(macrophage_columns %in% cell_columns)) stop("LM22 macrophage columns are missing.")
fractions$total_macrophages <- rowSums(fractions[, macrophage_columns])

exploratory <- cell_statistics(
  fractions %>% select(patient_id, all_of(cell_columns)),
  cohort,
  resamples,
  seed + 2000L
)
primary <- cell_statistics(
  fractions %>% select(patient_id, total_macrophages),
  cohort,
  resamples,
  seed + 4000L
) %>%
  mutate(group_fdr = NA_real_, correlation_fdr = NA_real_, prespecified = TRUE)
exploratory <- exploratory %>% mutate(prespecified = FALSE)
write_csv(bind_rows(primary, exploratory), file.path(output_dir, "lm22_statistics.csv"))
write_csv(fractions, file.path(output_dir, "lm22_normalized_fractions.csv"))

writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_session_info.txt"))
