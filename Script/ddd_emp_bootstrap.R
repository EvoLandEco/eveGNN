args <- commandArgs(TRUE)

lambda <- as.numeric(args[1])

mu <- as.numeric(args[2])

cap <- as.numeric(args[3])

ntip <- as.numeric(args[4])

family_name <- as.character(args[5])

tree_name <- as.character(args[6])

path <- as.character(args[7])

pars <- c(lambda, mu, cap)

meta <- c("Family" = family_name, "Tree" = tree_name)

boot_result <- eveGNN::tree_polymorphism_bootstrap(pars = pars,
                                                   age = 10,
                                                   ntip = ntip,
                                                   model = "DDD",
                                                   nrep = 100)

setwd(path)

for (i in 1:length(boot_result)) {
  saveRDS(boot_result[[i]], file = paste0("BOOT_", family_name, "_", tree_name, "_", i, ".rds"))
  eveGNN::export_to_gnn_bootstrap(data = boot_result[[i]],
                                  meta = meta,
                                  index = index,
                                  path = "EXPORT",
                                  undirected = FALSE)
}