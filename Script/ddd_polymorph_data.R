args <- commandArgs(TRUE)

name <- as.character(args[1])

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

nrep <- 100
age <- 10
ddmodel <- 1
nworkers_sim <- 4

# Create lists of lambda, mu, and cap values
lambda_list <- c(1.0, 1.5, 2.0, 2.5, 3.0)
mu_list <- c(0.2, 0.4, 0.6, 0.8)
cap_list <- c(200, 400, 600, 700)

# Create all possible combinations of lambda, mu, and cap values
param_list <- expand.grid(lambda = lambda_list, mu = mu_list, cap = cap_list)

# Create a list of parameters for the simulation
param_list <- purrr::transpose(param_list)

future::plan("multicore", workers = nworkers_sim)

ddd_poly_list <- furrr::future_map(param_list, function(params) {
  replicate(nrep, eveGNN::fixed_ddd_fixed_age(params, age = age, model = ddmodel), simplify = FALSE)
})

ddd_poly_list_all <- unlist(ddd_poly_list, recursive = FALSE)

ddd_poly_list_all <- purrr::transpose(ddd_poly_list_all)

if (!dir.exists("DDD_POLY_TES")) {
  dir.create("DDD_POLY_TES")
}

setwd("DDD_POLY_TES")

print("Exporting Polymorph TES Data to GNN")

eveGNN::export_to_gnn_with_params(ddd_poly_list_all, "tes", undirected = FALSE, master = FALSE)

setwd("..")

num_elements_to_sample <- length(ddd_poly_list_all$tes)

if (!dir.exists("DDD_MLE_TES")) {
  dir.create("DDD_MLE_TES")
}

setwd("DDD_MLE_TES")

print("Computing MLE for POLY TES")

if (!dir.exists("MLE_DATA")) {
  dir.create("MLE_DATA")
}

saveRDS(ddd_poly_list_all, paste0("MLE_DATA/ddd_mle_poly.rds"))

setwd("../../")

for (i in 1:num_elements_to_sample) {
  system(paste0("sbatch submit_ddd_pars_est_free_mle_poly.sh ", i, " ", name))
}