args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/ddd_sim.yaml")

if (!dir.exists(name)) {
  stop("Project folder does not exist")
}

setwd(name)

nrep <- params$nrep
proportion <- params$proportion
num_elements_to_sample <- ceiling(nrep * proportion)

if (!dir.exists("DDD_MLE_TES")) {
  stop("MLE data folder does not exist")
}

setwd("DDD_MLE_TES")

print("Re-Computing MLE for TES")

setwd("../../")

for (i in 1:num_elements_to_sample) {
  system(paste0("sbatch submit_ddd_pars_est_free_mle.sh ", i, " ", name))
}