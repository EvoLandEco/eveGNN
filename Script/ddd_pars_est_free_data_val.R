args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/ddd_sim.yaml")

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- params$dists
cap_range <- params$cap_range
max_mu <- params$max_mu
within_ranges <- params$within_ranges
nrep <- params$nrep
nrep <- nrep / 10
age <- params$age
ddmodel <- params$ddmodel
proportion <- params$proportion
nworkers_sim <- params$nworkers_sim
nworkers_mle <- params$nworkers_mle

future::plan("multicore", workers = nworkers_sim)

ddd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_ddd_fixed_age(dists,
                                                                                           cap_range = cap_range,
                                                                                           max_mu = max_mu,
                                                                                           age = age,
                                                                                           model = ddmodel), simplify = FALSE)

# Split list into training/testging data and validation (out-of-sample) data
ddd_list_all <- eveGNN::extract_by_range(tree_list = ddd_free_tes_list, ranges = within_ranges)

if (!dir.exists("DDD_FREE_TES_VAL")) {
  dir.create("DDD_FREE_TES_VAL")
}

setwd("DDD_FREE_TES_VAL")

print("Exporting Validation TES Data to GNN")

eveGNN::export_to_gnn_with_params(ddd_list_all$within_range, "tes", undirected = FALSE, master = FALSE)

