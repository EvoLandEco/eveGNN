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

# create 10 bins between 10 and 2000 nodes
bd_list_bin_10_200 <- list()
bd_list_bin_200_400 <- list()
bd_list_bin_400_600 <- list()
bd_list_bin_600_800 <- list()
bd_list_bin_800_1000 <- list()
bd_list_bin_1000_1200 <- list()
bd_list_bin_1200_1400 <- list()
bd_list_bin_1400_1600 <- list()
bd_list_bin_1600_1800 <- list()
bd_list_bin_1800_2000 <- list()

# Set length for each bin
bin_length <- ceiling(nrep / 10)

flag <- TRUE

while (flag == TRUE) {
  # First generate a batch of trees
  bd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_bd_fixed_age(dists, age = age), simplify = FALSE)

  # Fill in the bins
  for (i in 1:length(bd_free_tes_list)) {
    tree <- bd_free_tes_list[[i]]$tes
    n_nodes <- 2 * tree$Nnode + 1
    if (n_nodes >= 10 &&
      n_nodes < 200 &&
      length(bd_list_bin_10_200) < bin_length) {
      bd_list_bin_10_200[[length(bd_list_bin_10_200) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 200 &&
      n_nodes < 400 &&
      length(bd_list_bin_200_400) < bin_length) {
      bd_list_bin_200_400[[length(bd_list_bin_200_400) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 400 &&
      n_nodes < 600 &&
      length(bd_list_bin_400_600) < bin_length) {
      bd_list_bin_400_600[[length(bd_list_bin_400_600) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 600 &&
      n_nodes < 800 &&
      length(bd_list_bin_600_800) < bin_length) {
      bd_list_bin_600_800[[length(bd_list_bin_600_800) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 800 &&
      n_nodes < 1000 &&
      length(bd_list_bin_800_1000) < bin_length) {
      bd_list_bin_800_1000[[length(bd_list_bin_800_1000) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 1000 &&
      n_nodes < 1200 &&
      length(bd_list_bin_1000_1200) < bin_length) {
      bd_list_bin_1000_1200[[length(bd_list_bin_1000_1200) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 1200 &&
      n_nodes < 1400 &&
      length(bd_list_bin_1200_1400) < bin_length) {
      bd_list_bin_1200_1400[[length(bd_list_bin_1200_1400) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 1400 &&
      n_nodes < 1600 &&
      length(bd_list_bin_1400_1600) < bin_length) {
      bd_list_bin_1400_1600[[length(bd_list_bin_1400_1600) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 1600 &&
      n_nodes < 1800 &&
      length(bd_list_bin_1600_1800) < bin_length) {
      bd_list_bin_1600_1800[[length(bd_list_bin_1600_1800) + 1]] <- bd_free_tes_list[[i]]
    } else if (n_nodes >= 1800 &&
      n_nodes < 2000 &&
      length(bd_list_bin_1800_2000) < bin_length) {
      bd_list_bin_1800_2000[[length(bd_list_bin_1800_2000) + 1]] <- bd_free_tes_list[[i]]
    }
  }

  # check if all bins are filled
  if (length(bd_list_bin_10_200) == bin_length &&
    length(bd_list_bin_200_400) == bin_length &&
    length(bd_list_bin_400_600) == bin_length &&
    length(bd_list_bin_600_800) == bin_length &&
    length(bd_list_bin_800_1000) == bin_length &&
    length(bd_list_bin_1000_1200) == bin_length &&
    length(bd_list_bin_1200_1400) == bin_length &&
    length(bd_list_bin_1400_1600) == bin_length &&
    length(bd_list_bin_1600_1800) == bin_length &&
    length(bd_list_bin_1800_2000) == bin_length) {
    flag <- FALSE
  }
}

# Combine all bins into a single list
bd_rebalanced_list <- c(bd_list_bin_10_200, bd_list_bin_200_400, bd_list_bin_400_600, bd_list_bin_600_800, bd_list_bin_800_1000, bd_list_bin_1000_1200, bd_list_bin_1200_1400, bd_list_bin_1400_1600, bd_list_bin_1600_1800, bd_list_bin_1800_2000)

# Split list into training/testing data and validation (out-of-sample) data
bd_list_all <- eveGNN::extract_by_range(tree_list = bd_rebalanced_list, ranges = within_ranges)

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