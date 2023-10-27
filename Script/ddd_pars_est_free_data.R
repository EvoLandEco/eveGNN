args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)
dir.create("DDD_FREE_TES")
setwd("DDD_FREE_TES")

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 1.0),
  list(distribution = "uniform", n = 1, min = 0, max = 0.4)
)

cap_range <- c(10,1000)

ddd_free_tes_list <- replicate(20000, eveGNN::randomized_ddd_fixed_age(dists,
                                                                          cap_range = cap_range,
                                                                          age = 10,
                                                                          model = 1), simplify = FALSE)

ddd_free_tes_list <- purrr::transpose(ddd_free_tes_list)

eveGNN::export_to_gnn_with_params(ddd_free_tes_list, "tes")

ddd_free_tes_list_test <- eveGNN::get_test_data(ddd_free_tes_list, 0.025)
mean_diffs <- eveGNN::compute_accuracy_dd_ml_free(dists, cap_range, ddd_free_tes_list_test, strategy = "multicore", workers = 16)
saveRDS(mean_diffs, "mean_diffs_DDD_FREE_TES.rds")