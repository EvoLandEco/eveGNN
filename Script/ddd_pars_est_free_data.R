args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

params <- yaml::read_yaml("../Config/ddd_sim.yaml")

dists <- params$dists
cap_range <- params$cap_range
within_ranges <- params$within_ranges
nrep <- params$nrep
age <- params$age
ddmodel <- params$ddmodel
proportion <- params$proportion
nworkers_sim <- params$nworkers_sim
nworkers_mle <- params$nworkers_mle

future::plan("multicore", workers = nworkers_sim)

ddd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_ddd_fixed_age(dists,
                                                                        cap_range = cap_range,
                                                                        age = age,
                                                                        model = ddmodel), simplify = FALSE)

# Split list into training/testging data and validation (out-of-sample) data
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

num_elements_to_sample <- ceiling(length(ddd_free_tes_list) * proportion)

ddd_mle_list <- sample(ddd_free_tes_list, num_elements_to_sample)

ddd_mle_list <- purrr::transpose(ddd_mle_list)

if (!dir.exists("DDD_MLE_TES")) {
  dir.create("DDD_MLE_TES")
}

setwd("DDD_MLE_TES")

print("Computing MLE for TES")

ddd_mle_diffs_tes <- eveGNN::compute_accuracy_dd_ml_free(dists, cap_range, ddd_mle_list, strategy = "multicore", workers = nworkers_mle)

saveRDS(ddd_mle_diffs_tes, "mle_diffs_DDD_FREE_TES.rds")
