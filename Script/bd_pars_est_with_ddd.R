# export_to_gnn_with_params_bd_as_ddd <- function(data, which = "tas", undirected = FALSE) {
#   path <- file.path("GNN/tree/")
#   path_EL <- file.path("GNN/tree/EL/")
#   path_ST <- file.path("GNN/tree/ST/")
#   path_BT <- file.path("GNN/tree/BT/")
#   eve:::check_path(path)
#   eve:::check_path(path_EL)
#   eve:::check_path(path_ST)
#   eve:::check_path(path_BT)
#
#   if (which == "tas") {
#     for (i in seq_along(data$tas)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path, "/tree_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
#     }
#     for (i in seq_along(data$tas)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_adj_mat(data$tas[[i]]), file = file_name)
#     }
#     for (i in seq_along(data$tas)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path_ST, "/ST_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_stats(data$tas[[i]]), file = file_name)
#     }
#     for (i in seq_along(data$tas)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path_BT, "/BT_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_brts(data$tas[[i]]), file = file_name)
#     }
#   } else if (which == "tes") {
#     for (i in seq_along(data$tes)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path, "/tree_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
#     }
#     for (i in seq_along(data$tes)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_adj_mat(data$tes[[i]]), file = file_name)
#     }
#     for (i in seq_along(data$tes)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path_ST, "/ST_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_stats(data$tes[[i]]), file = file_name)
#     }
#     for (i in seq_along(data$tes)) {
#       la <- data$pars[[i]][1]
#       mu <- data$pars[[i]][2]
#       age <- data$age[[i]]
#       file_name <- paste0(path_BT, "/BT_", la, "_", mu, "_", 0, "_", age, "_", i, ".rds")
#       saveRDS(eveGNN::tree_to_brts(data$tes[[i]]), file = file_name)
#     }
#   }
# }
#
args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/bd_sim.yaml")

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

# dists <- params$dists
# within_ranges <- params$within_ranges
# nrep <- params$nrep
# age <- params$age
# proportion <- params$proportion
# nworkers_sim <- params$nworkers_sim
# nworkers_mle <- params$nworkers_mle
#
# future::plan("multicore", workers = nworkers_sim)
#
# bd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_bd_fixed_age(dists, age = age), simplify = FALSE)
#
# # Split list into training/testing data and validation (out-of-sample) data
# bd_list_all <- eveGNN::extract_by_range(tree_list = bd_free_tes_list, ranges = within_ranges)
#
# if (!dir.exists("DDD_FREE_TES")) {
#   dir.create("DDD_FREE_TES")
# }
#
# setwd("DDD_FREE_TES")
#
# print("Exporting Training/Testing TES Data to GNN")
#
# export_to_gnn_with_params_bd_as_ddd(bd_list_all$within_range, "tes", undirected = FALSE)
#
# setwd("..")
#
# num_elements_to_sample <- ceiling(length(bd_free_tes_list) * proportion)
#
# ddd_mle_list <- sample(bd_free_tes_list, num_elements_to_sample)
#
# ddd_mle_list <- purrr::transpose(ddd_mle_list)
#
# if (!dir.exists("DDD_MLE_TES")) {
#   dir.create("DDD_MLE_TES")
# }
#
# setwd("DDD_MLE_TES")
#
# print("Computing MLE for TES")
#
# if (!dir.exists("MLE_DATA")) {
#   dir.create("MLE_DATA")
# }
#
# saveRDS(ddd_mle_list, paste0("MLE_DATA/ddd_mle.rds"))
#
# setwd("../../")

num_elements_to_sample <- 2000

for (i in 1:num_elements_to_sample) {
  system(paste0("sbatch submit_ddd_pars_est_free_mle.sh ", i, " ", name))
}