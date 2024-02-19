args <- commandArgs(TRUE)

name <- as.character(args[1])

setwd(name)

future::plan("multicore", workers = 4)

bd_poly_list_1 <- future.apply::future_replicate(1000,
                                               eveGNN::bd_fixed_age(0.6, 0.1, age = 10),
                                               simplify = FALSE)

bd_poly_list_2 <- future.apply::future_replicate(1000,
                                               eveGNN::bd_fixed_age(0.6, 0.2, age = 10),
                                               simplify = FALSE)

bd_poly_list_3 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.6, 0.3, age = 10),
                                                simplify = FALSE)

bd_poly_list_4 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.6, 0.4, age = 10),
                                                simplify = FALSE)

bd_poly_list_5 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.6, 0.5, age = 10),
                                                simplify = FALSE)

bd_poly_list_6 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.5, 0.1, age = 10),
                                                simplify = FALSE)

bd_poly_list_7 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.4, 0.1, age = 10),
                                                simplify = FALSE)

bd_poly_list_8 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.3, 0.1, age = 10),
                                                simplify = FALSE)

bd_poly_list_9 <- future.apply::future_replicate(1000,
                                                eveGNN::bd_fixed_age(0.2, 0.1, age = 10),
                                                simplify = FALSE)

bd_poly_list <- c(bd_poly_list_1, bd_poly_list_2,
                  bd_poly_list_3, bd_poly_list_4,
                  bd_poly_list_5, bd_poly_list_6,
                  bd_poly_list_7, bd_poly_list_8,
                  bd_poly_list_9)

if (!dir.exists("BD_POLY_TES")) {
  dir.create("BD_POLY_TES")
}

setwd("BD_POLY_TES")

print("Exporting BD Poly TES Data to GNN")

bd_poly_list <- purrr::transpose(bd_poly_list)

eveGNN::export_to_gnn_with_params_bd(bd_poly_list, "tes", undirected = FALSE)

setwd("..")

if (!dir.exists("BD_POLY_MLE")) {
  dir.create("BD_POLY_MLE")
}

setwd("BD_POLY_MLE")

print("Computing MLE for BD Poly Data without initial parameters")

bd_mle_diffs_tes_no_init <- eveGNN::compute_accuracy_bd_ml_free_no_init(bd_poly_list, strategy = "multicore", workers = 8)

saveRDS(bd_mle_diffs_tes_no_init, "mle_diffs_BD_POLY_TES.rds")
