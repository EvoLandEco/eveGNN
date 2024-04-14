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