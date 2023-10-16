args <- commandArgs(TRUE)

name <- as.character(args[1])
cap <- as.numeric(args[2])
index <- as.character(args[3])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)
dir.create("DDD_TES")
setwd("DDD_TES")

ddd_tes_list <- replicate(500, eveGNN::dd_sim_fix_n(200, pars = c(c(0.6, 0.1), cap), 10, 1), simplify = FALSE)
ddd_tes_list <- purrr::transpose(ddd_tes_list)

eveGNN::export_to_gnn(ddd_tes_list, index, "tes")
DDD_params <- data.frame(cap = cap, lambda = 0.6, mu = 0.1, age = 10, n = 200)
write.table(DDD_params, paste0("DDD_TES_params.txt"))