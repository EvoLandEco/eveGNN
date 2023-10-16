args <- commandArgs(TRUE)

name <- args[1]

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)
dir.create("BD_TES")
setwd("BD_TES")

bd_tes <- replicate(500, ape::rphylo(n = 200, birth = 0.6, death = 0.1, T0 = 10, fossils = FALSE), simplify = FALSE)
bd_tes_list <- list(tes = bd_tes)

eveGNN::export_to_gnn(bd_tes_list, 1, "tes")
BD_params <- data.frame(lambda = 0.6, mu = 0.1, age = 10, n = 200)
write.table(BD_params, paste0("BD_TES_params.txt"))
