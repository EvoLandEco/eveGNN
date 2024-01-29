args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/pbd_sim.yaml")

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

pbd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_pbd_fixed_age(dists, age = age), simplify = FALSE)

# Split list into training/testging data and validation (out-of-sample) data
pbd_list_all <- eveGNN::extract_by_range(tree_list = pbd_free_tes_list, ranges = within_ranges)

if (!dir.exists("PBD_FREE_TES")) {
  dir.create("PBD_FREE_TES")
}

setwd("PBD_FREE_TES")

print("Exporting Training/Testing TES Data to GNN")

eveGNN::export_to_gnn_with_params_pbd(pbd_list_all$within_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("PBD_VAL_TES")) {
  dir.create("PBD_VAL_TES")
}

setwd("PBD_VAL_TES")

print("Exporting Validation TES Data to GNN")

eveGNN::export_to_gnn_with_params_pbd(pbd_list_all$outside_range, "tes", undirected = FALSE)

setwd("..")

num_elements_to_sample <- ceiling(length(pbd_free_tes_list) * proportion)

pbd_mle_list <- sample(pbd_free_tes_list, num_elements_to_sample)

pbd_mle_list <- purrr::transpose(pbd_mle_list)

if (!dir.exists("PBD_MLE_TES")) {
  dir.create("PBD_MLE_TES")
}

setwd("PBD_MLE_TES")

print("Computing MLE for TES")

pbd_mle_diffs_tes <- eveGNN::compute_accuracy_pbd_ml_free(pbd_mle_list, strategy = "multicore", workers = nworkers_mle)

if (!dir.exists("NO_INIT")) {
  dir.create("NO_INIT")
}

setwd("NO_INIT")

print("Computing MLE for TES without initial parameters")

pbd_mle_diffs_tes_no_init <- eveGNN::compute_accuracy_pbd_ml_free_no_init(pbd_mle_list, strategy = "multicore", workers = nworkers_mle)