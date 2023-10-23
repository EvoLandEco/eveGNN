args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)
dir.create("DDD_LAMU_TES")
setwd("DDD_LAMU_TES")

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 1.0),
  list(distribution = "uniform", n = 1, min = 0, max = 0.4)
)

ddd_lamu_tes_list <- eveGNN::batch_sim_ddd(dists, 100, 10, 1, 10000)

eveGNN::export_to_gnn_with_params(ddd_lamu_tes_list, "tes")

ddd_lamu_tes_list_test <- eveGNN::get_test_data(ddd_lamu_tes_list, 0.1)

mean_diffs <- eveGNN::compute_accuracy_dd_ml(dists, ddd_lamu_tes_list_test, strategy = "multicore", workers = 8)

saveRDS(mean_diffs, "mean_diffs_DDD_LAMU_TES.rds")