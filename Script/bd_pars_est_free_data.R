args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/bd_sim.yaml")

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- params$dists
within_ranges <- params$within_ranges
nrep <- params$nrep
age <- params$age
proportion <- params$proportion
nworkers_sim <- params$nworkers_sim
nworkers_mle <- params$nworkers_mle

future::plan("multicore", workers = nworkers_sim)

bd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_bd_fixed_age(dists, age = age), simplify = FALSE)

# Split list into training/testing data and validation (out-of-sample) data
bd_list_all <- eveGNN::extract_by_range(tree_list = bd_free_tes_list, ranges = within_ranges)

if (!dir.exists("BD_FREE_TES")) {
  dir.create("BD_FREE_TES")
}

setwd("BD_FREE_TES")

print("Exporting Training/Testing TES Data to GNN")

eveGNN::export_to_gnn_with_params_bd(bd_list_all$within_range, "tes", undirected = FALSE)

# setwd("..")
#
# if (!dir.exists("BD_FREE_TAS")) {
#   dir.create("BD_FREE_TAS")
# }
#
# setwd("BD_FREE_TAS")
#
# print("Exporting Training/Testing TAS Data to GNN")
#
# eveGNN::export_to_gnn_with_params_bd(bd_list_all$within_range, "tas", undirected = FALSE)
#
# setwd("..")
#
# if (!dir.exists("BD_VAL_TES")) {
#   dir.create("BD_VAL_TES")
# }
#
# setwd("BD_VAL_TES")
#
# print("Exporting Validation TES Data to GNN")
#
# eveGNN::export_to_gnn_with_params_bd(bd_list_all$outside_range, "tes", undirected = FALSE)
#
# setwd("..")
#
# if (!dir.exists("BD_VAL_TAS")) {
#   dir.create("BD_VAL_TAS")
# }
#
# setwd("BD_VAL_TAS")
#
# print("Exporting Validation TAS Data to GNN")
#
# eveGNN::export_to_gnn_with_params_bd(bd_list_all$outside_range, "tas", undirected = FALSE)

setwd("..")

num_elements_to_sample <- ceiling(length(bd_free_tes_list) * proportion)

bd_mle_list <- sample(bd_free_tes_list, num_elements_to_sample)

bd_mle_list <- purrr::transpose(bd_mle_list)

if (!dir.exists("BD_MLE_TES")) {
  dir.create("BD_MLE_TES")
}

setwd("BD_MLE_TES")

print("Computing MLE for TES")

bd_mle_diffs_tes <- eveGNN::compute_accuracy_bd_ml_free(bd_mle_list, strategy = "multicore", workers = nworkers_mle)

saveRDS(bd_mle_diffs_tes, "mle_diffs_BD_FREE_TES.rds")

if (!dir.exists("NO_INIT")) {
  dir.create("NO_INIT")
}

setwd("NO_INIT")

print("Computing MLE for TES without initial parameters")

bd_mle_diffs_tes_no_init <- eveGNN::compute_accuracy_bd_ml_free_no_init(bd_mle_list, strategy = "multicore", workers = nworkers_mle)

saveRDS(bd_mle_diffs_tes_no_init, "mle_diffs_BD_FREE_TES_NO_INIT.rds")