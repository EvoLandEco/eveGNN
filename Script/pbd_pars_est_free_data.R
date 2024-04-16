args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/pbd_sim.yaml")

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- params$dists
max_mu1 <- params$max_mu1
max_mu2 <- params$max_mu2
max_mus <- c(max_mu1, max_mu2)
within_ranges <- params$within_ranges
nrep <- params$nrep
age <- params$age
proportion <- params$proportion
nworkers_sim <- params$nworkers_sim
nworkers_mle <- params$nworkers_mle

future::plan("multicore", workers = nworkers_sim)

# create 10 bins between 10 and 2000 nodes
pbd_list_bin_10_200 <- list()
pbd_list_bin_200_400 <- list()
pbd_list_bin_400_600 <- list()
pbd_list_bin_600_800 <- list()
pbd_list_bin_800_1000 <- list()
pbd_list_bin_1000_1200 <- list()
pbd_list_bin_1200_1400 <- list()
pbd_list_bin_1400_1600 <- list()
pbd_list_bin_1600_1800 <- list()
pbd_list_bin_1800_2000 <- list()

# Set length for each bin
bin_length <- ceiling(nrep / 10)

flag <- TRUE

while (flag == TRUE) {
  # First generate a batch of trees
  pbd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_pbd_fixed_age(dists, max_mus = max_mus, age = age), simplify = FALSE)

  # Fill in the bins
  for (i in 1:length(pbd_free_tes_list)) {
    tree <- pbd_free_tes_list[[i]]$tes
    n_nodes <- 2 * tree$Nnode + 1
    if (n_nodes >= 10 &&
      n_nodes < 200 &&
      length(pbd_list_bin_10_200) < bin_length) {
      pbd_list_bin_10_200[[length(pbd_list_bin_10_200) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 200 &&
      n_nodes < 400 &&
      length(pbd_list_bin_200_400) < bin_length) {
      pbd_list_bin_200_400[[length(pbd_list_bin_200_400) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 400 &&
      n_nodes < 600 &&
      length(pbd_list_bin_400_600) < bin_length) {
      pbd_list_bin_400_600[[length(pbd_list_bin_400_600) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 600 &&
      n_nodes < 800 &&
      length(pbd_list_bin_600_800) < bin_length) {
      pbd_list_bin_600_800[[length(pbd_list_bin_600_800) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 800 &&
      n_nodes < 1000 &&
      length(pbd_list_bin_800_1000) < bin_length) {
      pbd_list_bin_800_1000[[length(pbd_list_bin_800_1000) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 1000 &&
      n_nodes < 1200 &&
      length(pbd_list_bin_1000_1200) < bin_length) {
      pbd_list_bin_1000_1200[[length(pbd_list_bin_1000_1200) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 1200 &&
      n_nodes < 1400 &&
      length(pbd_list_bin_1200_1400) < bin_length) {
      pbd_list_bin_1200_1400[[length(pbd_list_bin_1200_1400) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 1400 &&
      n_nodes < 1600 &&
      length(pbd_list_bin_1400_1600) < bin_length) {
      pbd_list_bin_1400_1600[[length(pbd_list_bin_1400_1600) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 1600 &&
      n_nodes < 1800 &&
      length(pbd_list_bin_1600_1800) < bin_length) {
      pbd_list_bin_1600_1800[[length(pbd_list_bin_1600_1800) + 1]] <- pbd_free_tes_list[[i]]
    } else if (n_nodes >= 1800 &&
      n_nodes < 2000 &&
      length(pbd_list_bin_1800_2000) < bin_length) {
      pbd_list_bin_1800_2000[[length(pbd_list_bin_1800_2000) + 1]] <- pbd_free_tes_list[[i]]
    }
  }

  # Check if all bins are filled
  if (length(pbd_list_bin_10_200) == bin_length &&
    length(pbd_list_bin_200_400) == bin_length &&
    length(pbd_list_bin_400_600) == bin_length &&
    length(pbd_list_bin_600_800) == bin_length &&
    length(pbd_list_bin_800_1000) == bin_length &&
    length(pbd_list_bin_1000_1200) == bin_length &&
    length(pbd_list_bin_1200_1400) == bin_length &&
    length(pbd_list_bin_1400_1600) == bin_length &&
    length(pbd_list_bin_1600_1800) == bin_length &&
    length(pbd_list_bin_1800_2000) == bin_length) {
    flag <- FALSE
  }
}

# Combine all bins into a single list
pbd_rebalanced_list <- c(pbd_list_bin_10_200, pbd_list_bin_200_400, pbd_list_bin_400_600, pbd_list_bin_600_800, pbd_list_bin_800_1000, pbd_list_bin_1000_1200, pbd_list_bin_1200_1400, pbd_list_bin_1400_1600, pbd_list_bin_1600_1800, pbd_list_bin_1800_2000)

pbd_list_all <- eveGNN::extract_by_range(tree_list = pbd_rebalanced_list, ranges = within_ranges)

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