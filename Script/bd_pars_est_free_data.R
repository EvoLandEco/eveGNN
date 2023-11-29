args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 0.8),
  list(distribution = "uniform", n = 1, min = 0.0, max = 0.2)
)

future::plan("multicore", workers = 4)

bd_free_tes_list <- future.apply::future_replicate(50000, eveGNN::randomized_bd_fixed_age(dists, age = 10), simplify = FALSE)

# Split list into training/testging data and validation (out-of-sample) data
within_ranges <- list(c(0.52, 0.78), c(0.02, 0.18))
bd_list_all <- extract_by_range(tree_list = bd_free_tes_list, ranges = within_ranges)

if (!dir.exists("BD_FREE_TES")) {
  dir.create("BD_FREE_TES")
}

setwd("BD_FREE_TES")

print("Exporting Training/Testing TES Data to GNN")

eveGNN::export_to_gnn_with_params(bd_list_all$within_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("BD_FREE_TAS")) {
  dir.create("BD_FREE_TAS")
}

setwd("BD_FREE_TAS")

print("Exporting Training/Testing TAS Data to GNN")

eveGNN::export_to_gnn_with_params(bd_list_all$within_range, "tas", undirected = FALSE)

setwd("..")

if (!dir.exists("BD_VAL_TES")) {
  dir.create("BD_VAL_TES")
}

setwd("BD_VAL_TES")

print("Exporting Validation TES Data to GNN")

eveGNN::export_to_gnn_with_params(bd_list_all$outside_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("BD_VAL_TAS")) {
  dir.create("BD_VAL_TAS")
}

setwd("BD_VAL_TAS")

print("Exporting Validation TAS Data to GNN")

eveGNN::export_to_gnn_with_params(bd_list_all$outside_range, "tas", undirected = FALSE)

setwd("..")

proportion <- 0.005

num_elements_to_sample <- ceiling(length(bd_free_tes_list) * proportion)

bd_mle_list <- sample(bd_free_tes_list, num_elements_to_sample)

bd_mle_list <- purrr::transpose(bd_mle_list)

if (!dir.exists("BD_MLE_TES")) {
  dir.create("BD_MLE_TES")
}

setwd("BD_MLE_TES")

print("Computing MLE for TES")

bd_mle_diffs_tes <- eveGNN::compute_accuracy_bd_ml_free(dists, bd_mle_list, strategy = "multicore", workers = 16)

saveRDS(bd_mle_diffs_tes, "mle_diffs_BD_FREE_TES.rds")