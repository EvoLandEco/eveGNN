args <- commandArgs(TRUE)

name <- as.character(args[1])

data <- readRDS(file.path(name, "EMP_DATA/OTT_trees.rds"))

setwd(name)

num_elements <- length(data)

if (!dir.exists("DDD_EMP_MLE")) {
  dir.create("DDD_EMP_MLE")
}

setwd("../")

for (i in 1:num_elements) {
  system(paste0("sbatch submit_ddd_pars_est_ott_mle.sh ", i, " ", name))
}