args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)
dir.create("EVE_FREE_TES")
setwd("EVE_FREE_TES")

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 0.8),
  list(distribution = "uniform", n = 1, min = 0, max = 0.3),
  list(distribution = "uniform", n = 1, min = -0.02, max = 0),
  list(distribution = "uniform", n = 1, min = -0.002, max = 0)
)

eve_free_tes_pd_list <- replicate(5000, eveGNN::randomized_eve_fixed_age(dists, age = 8,
                                                                model = "dsce2",
                                                                metric = "pd", offset = "simtime"), simplify = FALSE)
eve_free_tes_pd_list <- purrr::transpose(eve_free_tes_pd_list)
eveGNN::export_to_gnn_with_params_eve(eve_free_tes_pd_list, "tes")

eve_free_tes_ed_list <- replicate(5000, eveGNN::randomized_eve_fixed_age(dists, age = 8,
                                                                model = "dsce2",
                                                                metric = "ed", offset = "none"), simplify = FALSE)
eve_free_tes_ed_list <- purrr::transpose(eve_free_tes_ed_list)
eveGNN::export_to_gnn_with_params_eve(eve_free_tes_ed_list, "tes")

eve_free_tes_nnd_list <- replicate(5000, eveGNN::randomized_eve_fixed_age(dists, age = 8,
                                                                 model = "dsce2",
                                                                 metric = "nnd", offset = "none"), simplify = FALSE)
eve_free_tes_nnd_list <- purrr::transpose(eve_free_tes_nnd_list)
eveGNN::export_to_gnn_with_params_eve(eve_free_tes_nnd_list, "tes")