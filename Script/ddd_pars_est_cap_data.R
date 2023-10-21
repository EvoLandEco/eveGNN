args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)
dir.create("DDD_CAP_TES")
setwd("DDD_CAP_TES")

cap_range <- c(10,1000)

ddd_cap_tes_list <- replicate(3000, eveGNN::randomized_ddd_fixed_la_mu_age(cap_range = cap_range,
                                                                     la = 0.6,
                                                                     mu = 0.1,
                                                                     age = 10,
                                                                     model = 1), simplify = FALSE)

ddd_cap_tes_list <- purrr::transpose(ddd_cap_tes_list)

eveGNN::export_to_gnn_with_params(ddd_cap_tes_list, name, "tes")