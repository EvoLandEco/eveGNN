args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 0.9),
  list(distribution = "uniform", n = 1, min = 0, max = 0.4),
  list(distribution = "uniform", n = 1, min = -0.03, max = 0),
  list(distribution = "uniform", n = 1, min = -0.003, max = 0)
)

future::plan("multicore", workers = 6)

eve_free_pd_list <- future.apply::future_replicate(50000, eveGNN::randomized_eve_fixed_age(dists, age = 8,
                                                                         model = "dsce2",
                                                                         metric = "pd", offset = "simtime"), simplify = FALSE)

eve_free_ed_list <- future.apply::future_replicate(50000, eveGNN::randomized_eve_fixed_age(dists, age = 8,
                                                                         model = "dsce2",
                                                                         metric = "ed", offset = "none"), simplify = FALSE)

eve_free_nnd_list <- future.apply::future_replicate(50000, eveGNN::randomized_eve_fixed_age(dists, age = 8,
                                                                          model = "dsce2",
                                                                          metric = "nnd", offset = "none"), simplify = FALSE)

within_ranges <- list(c(0.52, 0.88), c(0.02, 0.38), c(-0.028, -0.002), c(-0.0028, -0.0002))
eve_pd_list_all <- eveGNN::extract_by_range(tree_list = eve_free_pd_list, ranges = within_ranges)
eve_ed_list_all <- eveGNN::extract_by_range(tree_list = eve_free_ed_list, ranges = within_ranges)
eve_nnd_list_all <- eveGNN::extract_by_range(tree_list = eve_free_nnd_list, ranges = within_ranges)

if (!dir.exists("EVE_FREE_TES")) {
  dir.create("EVE_FREE_TES")
}

setwd("EVE_FREE_TES")

print("Exporting Training/Testing TES Data to GNN")

eveGNN::export_to_gnn_with_params(eve_pd_list_all$within_range, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_ed_list_all$within_range, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_nnd_list_all$within_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("EVE_FREE_TAS")) {
  dir.create("EVE_FREE_TAS")
}

setwd("EVE_FREE_TAS")

print("Exporting Training/Testing TAS Data to GNN")

eveGNN::export_to_gnn_with_params(eve_pd_list_all$within_range, "tas", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_ed_list_all$within_range, "tas", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_nnd_list_all$within_range, "tas", undirected = FALSE)

setwd("..")

if (!dir.exists("EVE_VAL_TES")) {
  dir.create("EVE_VAL_TES")
}

setwd("EVE_VAL_TES")

print("Exporting Validation TES Data to GNN")

eveGNN::export_to_gnn_with_params(eve_pd_list_all$outside_range, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_ed_list_all$outside_range, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_nnd_list_all$outside_range, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("EVE_VAL_TAS")) {
  dir.create("EVE_VAL_TAS")
}

setwd("EVE_VAL_TAS")

print("Exporting Validation TAS Data to GNN")

eveGNN::export_to_gnn_with_params(eve_pd_list_all$outside_range, "tas", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_ed_list_all$outside_range, "tas", undirected = FALSE)
eveGNN::export_to_gnn_with_params(eve_nnd_list_all$outside_range, "tas", undirected = FALSE)

