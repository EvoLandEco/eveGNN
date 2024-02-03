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

boot_result <- replicate(1000, {
  tree <- DDD::dd_sim(pars = pars, age = 10, ddmodel = 1)
}, simplify = FALSE)

setwd(path)

for (i in 1:length(boot_result)) {
  eveGNN::export_to_gnn_bootstrap(data = boot_result[[i]]$tes,
                                  meta = meta,
                                  index = i,
                                  path = "EXPORT",
                                  undirected = FALSE)
}