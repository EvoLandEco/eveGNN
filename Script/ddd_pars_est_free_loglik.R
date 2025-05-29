args <- commandArgs(TRUE)

name <- as.character(args[1])

for (i in 1:2000) {
  system(paste0("sbatch submit_ddd_pars_est_free_mle_loglik.sh ", i, " ", name))
}