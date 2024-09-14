args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/eve_sim.yaml")

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists_pd <- params$dists_pd
dists_ed <- params$dists_ed
dists_nnd <- params$dists_nnd
# within_ranges_pd <- params$within_ranges_pd
# within_ranges_ed <- params$within_ranges_ed
# within_ranges_nnd <- params$within_ranges_nnd
nrep <- params$nrep
age <- params$age
nworkers_sim <- params$nworkers_sim

future::plan("multicore", workers = nworkers_sim)

eve_free_pd_list <- future.apply::future_replicate(nrep, eveGNN::randomized_eve_fixed_age(dists_pd, age = age,
                                                                         metric = "pd", offset = "simtime"), simplify = FALSE)

eve_free_ed_list <- future.apply::future_replicate(nrep, eveGNN::randomized_eve_fixed_age(dists_ed, age = age,
                                                                         metric = "ed", offset = "none"), simplify = FALSE)

eve_free_nnd_list <- future.apply::future_replicate(nrep, eveGNN::randomized_eve_fixed_age(dists_nnd, age = age,
                                                                          metric = "nnd", offset = "none"), simplify = FALSE)

# eve_pd_list_all <- eveGNN::extract_by_range(tree_list = eve_free_pd_list, ranges = within_ranges_pd)
# eve_ed_list_all <- eveGNN::extract_by_range(tree_list = eve_free_ed_list, ranges = within_ranges_ed)
# eve_nnd_list_all <- eveGNN::extract_by_range(tree_list = eve_free_nnd_list, ranges = within_ranges_nnd)

# utility function to filter out NULL elements
remove_nulls <- function(lst) {
  # filter out all sublists with sublist$tes == NULL
  lst <- lst[sapply(lst, function(x) !is.null(x$tes))]
}

eve_pd_list_all <- remove_nulls(eve_free_pd_list)
eve_ed_list_all <- remove_nulls(eve_free_ed_list)
eve_nnd_list_all <- remove_nulls(eve_free_nnd_list)

# transpose all the lists
eve_pd_list_all <- purrr::transpose(eve_pd_list_all)
eve_ed_list_all <- purrr::transpose(eve_ed_list_all)
eve_nnd_list_all <- purrr::transpose(eve_nnd_list_all)

if (!dir.exists("EVE_FREE_TES")) {
  dir.create("EVE_FREE_TES")
}

setwd("EVE_FREE_TES")

print("Exporting Training/Testing TES Data to GNN")

# eveGNN::export_to_gnn_with_params_eve(eve_pd_list_all$within_range, "tes", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all$within_range, "tes", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all$within_range, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_pd_list_all, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all, "tes", undirected = FALSE)

# setwd("..")
#
# if (!dir.exists("EVE_FREE_TAS")) {
#   dir.create("EVE_FREE_TAS")
# }
#
# setwd("EVE_FREE_TAS")
#
# print("Exporting Training/Testing TAS Data to GNN")
#
# eveGNN::export_to_gnn_with_params_eve(eve_pd_list_all$within_range, "tas", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all$within_range, "tas", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all$within_range, "tas", undirected = FALSE)
#
# setwd("..")
#
# if (!dir.exists("EVE_VAL_TES")) {
#   dir.create("EVE_VAL_TES")
# }
#
# setwd("EVE_VAL_TES")
#
# print("Exporting Validation TES Data to GNN")
#
# eveGNN::export_to_gnn_with_params_eve(eve_pd_list_all$outside_range, "tes", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all$outside_range, "tes", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all$outside_range, "tes", undirected = FALSE)
#
# setwd("..")
#
# if (!dir.exists("EVE_VAL_TAS")) {
#   dir.create("EVE_VAL_TAS")
# }
#
# setwd("EVE_VAL_TAS")
#
# print("Exporting Validation TAS Data to GNN")
#
# eveGNN::export_to_gnn_with_params_eve(eve_pd_list_all$outside_range, "tas", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all$outside_range, "tas", undirected = FALSE)
# eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all$outside_range, "tas", undirected = FALSE)
