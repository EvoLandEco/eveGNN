args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

if (!dir.exists("PBD_FREE_TES")) {
  dir.create("PBD_FREE_TES")
}

setwd("PBD_FREE_TES")

dists <- list(
  list(distribution = "uniform", n = 1, min = 0.6, max = 1.2),
  list(distribution = "uniform", n = 1, min = 0.6, max = 1.0),
  list(distribution = "uniform", n = 1, min = 0.0, max = 0.3),
  list(distribution = "uniform", n = 1, min = 0.0, max = 0.3),
  list(distribution = "uniform", n = 1, min = 0.0, max = 0.3)
)

future::plan("multicore", workers = 16)

pbd_free_tes_list <- future.apply::future_replicate(50000, eveGNN::randomized_pbd_fixed_age(dists, age = 10), simplify = FALSE)
pbd_free_tes_list <- purrr::transpose(pbd_free_tes_list)

eveGNN::export_to_gnn_with_params_pbd(pbd_free_tes_list, "tes", undirected = FALSE)

# pbd_free_tes_list_test <- eveGNN::get_test_data(pbd_free_tes_list, 0.005)
#
# if (!dir.exists("MLE")) {
#   dir.create("MLE")
# }
#
# setwd("MLE")
#
# test_pbd_ml <- PBD::pbd_ML(brts = pbd_free_tes_list$brts[[1]],
#        idparsopt = c(1, 2, 3),
#        exteq = 1,
#        btorph = 0,
#        soc = 2,
#       # cond = 0,
#        verbose = FALSE)