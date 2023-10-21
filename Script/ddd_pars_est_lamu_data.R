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

ddd_lamu_tes_list <- eveGNN::batch_sim_ddd(dists, 100, 10, 1, 5000)

eveGNN::export_to_gnn_with_params(ddd_lamu_tes_list, name, "tes")