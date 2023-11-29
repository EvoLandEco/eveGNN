args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 1.0),
  list(distribution = "uniform", n = 1, min = 0, max = 0.4)
)

cap_range <- c(10,1000)

future::plan("multicore", workers = 4)

ddd_free_tes_list <- future.apply::future_replicate(50000, eveGNN::randomized_ddd_fixed_age(dists,
                                                                        cap_range = cap_range,
                                                                        age = 10,
                                                                        model = 1), simplify = FALSE)

# Split list into training/testging data and validation (out-of-sample) data
within_ranges <- list(c(0.52, 0.98), c(0.02, 0.38), c(100, 900))
ddd_list_all <- eveGNN::extract_by_range(tree_list = ddd_free_tes_list, ranges = within_ranges)

if (!dir.exists("DDD_FREE_TES")) {
  dir.create("DDD_FREE_TES")
}

setwd("DDD_FREE_TES")

print("Exporting Training/Testing TES Data to GNN")

eveGNN::export_to_gnn_with_params(ddd_list_all$within_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("DDD_FREE_TAS")) {
  dir.create("DDD_FREE_TAS")
}

setwd("DDD_FREE_TAS")

print("Exporting Training/Testing TAS Data to GNN")

eveGNN::export_to_gnn_with_params(ddd_list_all$within_range, "tas", undirected = FALSE)

setwd("..")

if (!dir.exists("DDD_VAL_TES")) {
  dir.create("DDD_VAL_TES")
}

setwd("DDD_VAL_TES")

print("Exporting Validation TES Data to GNN")

eveGNN::export_to_gnn_with_params(ddd_list_all$outside_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("DDD_VAL_TAS")) {
  dir.create("DDD_VAL_TAS")
}

setwd("DDD_VAL_TAS")

print("Exporting Validation TAS Data to GNN")

eveGNN::export_to_gnn_with_params(ddd_list_all$outside_range, "tas", undirected = FALSE)

setwd("..")

proportion <- 0.005

num_elements_to_sample <- ceiling(length(ddd_free_tes_list) * proportion)

ddd_mle_list <- sample(ddd_free_tes_list, num_elements_to_sample)

ddd_mle_list <- purrr::transpose(ddd_mle_list)

if (!dir.exists("DDD_MLE_TES")) {
  dir.create("DDD_MLE_TES")
}

setwd("DDD_MLE_TES")

print("Computing MLE for TES")

ddd_mle_diffs_tes <- eveGNN::compute_accuracy_dd_ml_free(dists, cap_range, ddd_mle_list, strategy = "multicore", workers = 16)

saveRDS(ddd_mle_diffs_tes, "mle_diffs_DDD_FREE_TES.rds")
