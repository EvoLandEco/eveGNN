args <- commandArgs(TRUE)

name <- as.character(args[1])

setwd(name)

condamine_list <- readRDS("EMP_DATA/condamine_list.rds")

for (i in 1:nrow(condamine_list)) {
    family_name <- condamine_list$Family[i]
    tree_name <- condamine_list$Tree[i]
    lambda <- condamine_list$lambda[i]
    mu <- condamine_list$mu[i]
    cap <- condamine_list$cap[i]
    system(paste0("sbatch ../submit_ddd_emp_mle_uncertainty.sh ",
                  paste0(family_name, " ", tree_name, " ", lambda, " ", mu, " ", cap)))
}
