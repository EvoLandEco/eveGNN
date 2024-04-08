args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/ddd_sim.yaml")

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists <- params$dists
cap_range <- params$cap_range
max_mu <- params$max_mu
within_ranges <- params$within_ranges
nrep <- 2000
age <- params$age
ddmodel <- params$ddmodel
nworkers_sim <- params$nworkers_sim

future::plan("multicore", workers = nworkers_sim)

ddd_free_tes_list <- future.apply::future_replicate(nrep, eveGNN::randomized_ddd_fixed_age(dists,
                                                                                           cap_range = cap_range,
                                                                                           max_mu = max_mu,
                                                                                           age = age,
                                                                                           model = ddmodel), simplify = FALSE)

ddd_list_all <- purrr::transpose(ddd_free_tes_list)

# Compute all summary statistics from tes
ddd_summary_stats <- furrr::future_map(ddd_list_all$tes, eveGNN:::tree_to_stats)

# Flatten the lists of pars to a data frame
pars_cca <- ddd_list_all$pars %>% purrr::transpose()
names(pars_cca) <- c("lambda", "mu", "K")
pars_cca <- data.frame(lapply(pars_cca, function(sublist) {
  unlist(sublist, use.names = FALSE)
}))

# Flatten the lists of stats to a data frame
stats_cca <- ddd_summary_stats %>% dplyr::bind_rows()

# Save the data
if (!dir.exists("CORR")) {
  dir.create("CORR")
}

setwd("CORR")

saveRDS(pars_cca, file = "pars.rds")
saveRDS(stats_cca, file = "stats.rds")

# Combine the parameters and stats data for correlation analysis
combined_data <- dplyr::bind_cols(pars_cca, stats_cca)

# Calculate correlations between parameters and stats
cor_matrix <- cor(combined_data)

# Extract correlations for parameters vs. stats
param_names <- names(pars_cca)
stats_names <- names(stats_cca)
cor_params_stats <- cor_matrix[param_names, stats_names]

# Calculate absolute values of correlations for heatmap
abs_cor_params_stats <- abs(cor_params_stats)

heatmap(abs_cor_params_stats, Rowv = NA, margins = c(9, 1))

# Save the correlation matrix
saveRDS(abs_cor_params_stats, file = "abs_cor_params_stats.rds")

