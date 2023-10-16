args <- commandArgs(TRUE)

name <- args[1]

dir.create(name)
setwd(name)
dir.create("BD_TAS")
setwd("BD_TAS")

bd_tas <- replicate(3000, ape::rlineage(birth = 0.6, death = 0.1, Tmax = 10), simplify = FALSE)
bd_tas_list <- list(tas = bd_tas)

export_to_gnn(bd_tas_list, 1, "tas")
BD_params <- data.frame(lambda = 0.6, mu = 0.1, age = 10)
write.table(BD_params, paste0("BD_TAS_params.txt"))

setwd("..")
dir.create("DDD_TAS")
setwd("DDD_TAS")

dists_bd <- list(
  list(distribution = "uniform", n = 1, min = 0.6, max = 0.6),
  list(distribution = "uniform", n = 1, min = 0.1, max = 0.1)
)

ddd_list <- list()
j <- 1
for (i in seq(from = 100, to = 600, by = 100)) {
  ddd_list[[j]] <- batch_sim_ddd(dists = dists_bd, cap = i, 10, 1, 3000)
  j <- j + 1
}

for (i in 1:length(ddd_list)) {
  export_to_gnn(ddd_list[[i]], i, "tas")
}

DDD_params <- data.frame(cap = seq(from = 100, to = 600, by = 100), lambda = 0.6, mu = 0.1, age = 10)
write.table(DDD_params, paste0("DDD_TAS_params.txt"))