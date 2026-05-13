args <- commandArgs(TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript eve_pars_est_bd_ed_nnd_data.R <name> [config_path]")
}

name <- as.character(args[1])
config_path <- if (length(args) >= 2) as.character(args[2]) else "../Config/eve_sim.yaml"

params <- yaml::read_yaml(config_path)

if (!dir.exists(name)) {
  dir.create(name, recursive = TRUE)
}
setwd(name)

# Use the existing dists_pd lambda range for the BD baseline to avoid adding a
# second simulation-range file. ED and NND use the existing eve ranges.
dists_bd <- params$dists_bd
if (is.null(dists_bd)) {
  dists_bd <- params$dists_pd
}
dists_ed <- params$dists_ed
dists_nnd <- params$dists_nnd

# Match the current 2-parameter experiment: set all gamma effects to zero.
# Leave beta ranges unchanged for ED/NND.
zero_gamma_effects <- function(dists) {
  dists[[4]]$max <- 0
  dists[[4]]$min <- 0
  dists[[5]]$max <- 0
  dists[[5]]$min <- 0
  dists
}
dists_ed <- zero_gamma_effects(dists_ed)
dists_nnd <- zero_gamma_effects(dists_nnd)

nrep <- params$nrep
age <- params$age
nworkers_sim <- params$nworkers_sim
if (is.null(nworkers_sim)) nworkers_sim <- 1
size_limit <- params$size_limit
if (is.null(size_limit)) size_limit <- 2000
min_ntip <- params$min_ntip
if (is.null(min_ntip)) min_ntip <- 10

future::plan("multicore", workers = nworkers_sim)
RcppParallel::setThreadOptions(numThreads = 1)

# Local BD simulator that exports in the same six-parameter EVE filename format:
# pars = lambda, mu, beta_n, beta_phi, gamma_n, gamma_phi.
randomized_bd_as_eve_fixed_age <- function(dists, age, min_ntip = 10, size_limit = 2000, max_attempts = 200) {
  for (attempt in seq_len(max_attempts)) {
    lambda <- eveGNN::generate_params(list(dists[[1]]))[[1]]
    mu <- runif(1, min = 0, max = 0.8 * lambda)

    tas <- tryCatch({
      ape::rlineage(birth = lambda, death = mu, Tmax = age)
    }, error = function(e) NULL)
    if (is.null(tas)) next

    tes <- tryCatch({
      ape::drop.fossil(tas)
    }, error = function(e) NULL)
    if (is.null(tes)) next

    ntip <- ape::Ntip(tes)
    if (ntip >= min_ntip && ntip <= size_limit) {
      result <- list()
      result$tes <- tes
      result$pars <- c(lambda, mu, 0, 0, 0, 0)
      result$age <- age
      result$model <- "BD"
      result$metric <- "bd"
      result$offset <- "none"
      return(result)
    }
  }
  list(tes = NULL)
}

remove_nulls <- function(lst) {
  lst[sapply(lst, function(x) !is.null(x$tes))]
}

print("Simulating BD baseline trees...")
eve_bd_list <- future.apply::future_replicate(
  nrep,
  randomized_bd_as_eve_fixed_age(dists_bd, age = age, min_ntip = min_ntip, size_limit = size_limit),
  simplify = FALSE
)

print("Simulating ED trees...")
eve_ed_list <- future.apply::future_replicate(
  nrep,
  eveGNN::randomized_eve_fixed_age(dists_ed, age = age, metric = "ed", offset = "none"),
  simplify = FALSE
)

print("Simulating NND trees...")
eve_nnd_list <- future.apply::future_replicate(
  nrep,
  eveGNN::randomized_eve_fixed_age(dists_nnd, age = age, metric = "nnd", offset = "none"),
  simplify = FALSE
)

eve_bd_list_all <- purrr::transpose(remove_nulls(eve_bd_list))
eve_ed_list_all <- purrr::transpose(remove_nulls(eve_ed_list))
eve_nnd_list_all <- purrr::transpose(remove_nulls(eve_nnd_list))

if (!dir.exists("EVE_FREE_TES")) {
  dir.create("EVE_FREE_TES")
}
setwd("EVE_FREE_TES")

print("Exporting BD/ED/NND Training/Testing TES Data to GNN folders")
eveGNN::export_to_gnn_with_params_eve(eve_bd_list_all, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all, "tes", undirected = FALSE)

setwd("..")
if (!dir.exists("EVE_VAL_TES")) {
  dir.create("EVE_VAL_TES")
}

print("Simulation/export complete.")
