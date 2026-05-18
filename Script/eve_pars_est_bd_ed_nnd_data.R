# Simulate and export BD / ED / NND eve trees for the TreeTransformer workflow.
# Version: 2026-05-18-hotfix2
# Changes vs v1:
#   - avoids requiring the 'future' package by default;
#   - uses explicit package namespaces for non-base/recommended calls;
#   - adds dependency preflight with clear error messages;
#   - supports serial, mclapply, or future backends via EVE_PARALLEL_BACKEND;
#   - accepts optional task_type as third CLI argument.

args <- base::commandArgs(TRUE)

if (base::length(args) < 1) {
  base::stop("Usage: Rscript eve_pars_est_bd_ed_nnd_data_v2.R <name> [config_path] [task_type]")
}

name <- base::as.character(args[1])
config_path <- if (base::length(args) >= 2) base::as.character(args[2]) else "../Config/eve_sim.yaml"
task_type <- if (base::length(args) >= 3) base::as.character(args[3]) else "EVE_FREE_TES"

require_namespace <- function(pkg, install_hint = TRUE) {
  if (!base::requireNamespace(pkg, quietly = TRUE)) {
    hint <- if (install_hint) {
      paste0(
        "\nInstall it in the R library visible to this SLURM job, or rerun the launcher with INSTALL_R_PKGS=1. ",
        "Current .libPaths():\n  ", paste(base::.libPaths(), collapse = "\n  ")
      )
    } else {
      ""
    }
    base::stop("Required R package is not available: '", pkg, "'.", hint, call. = FALSE)
  }
  TRUE
}

# Core package requirements. 'parallel' is a recommended R package and is used
# with explicit parallel:: calls when nworkers_sim > 1.
base::invisible(base::lapply(c("yaml", "ape", "eveGNN"), require_namespace))

# evesim is used by eveGNN in many installations; checking it here catches
# a common SLURM-library-path problem before the simulation starts.
if (!base::requireNamespace("evesim", quietly = TRUE)) {
  base::message("Warning: package 'evesim' is not directly visible. Continuing because some eveGNN builds may vendor or indirectly load it.")
}

base::message("[R] version: ", base::paste(base::R.Version()[c("major", "minor")], collapse = "."))
base::message("[R] .libPaths():\n  ", base::paste(base::.libPaths(), collapse = "\n  "))
base::message("[R] config_path: ", config_path)
base::message("[R] task_type: ", task_type)

params <- yaml::read_yaml(config_path)

if (!base::dir.exists(name)) {
  base::dir.create(name, recursive = TRUE)
}
base::setwd(name)

# Use the existing dists_pd lambda range for the BD baseline to avoid adding a
# second simulation-range file. ED and NND use the existing eve ranges.
dists_bd <- params$dists_bd
if (base::is.null(dists_bd)) {
  dists_bd <- params$dists_pd
}
dists_ed <- params$dists_ed
dists_nnd <- params$dists_nnd

if (base::is.null(dists_bd) || base::is.null(dists_ed) || base::is.null(dists_nnd)) {
  base::stop("Config must define dists_ed and dists_nnd, and either dists_bd or dists_pd.")
}

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
if (base::is.null(nworkers_sim)) nworkers_sim <- 1
size_limit <- params$size_limit
if (base::is.null(size_limit)) size_limit <- 2000
min_ntip <- params$min_ntip
if (base::is.null(min_ntip)) min_ntip <- 10

seed <- params$seed
if (base::is.null(seed)) {
  env_seed <- base::Sys.getenv("EVE_SIM_SEED", unset = "")
  seed <- if (env_seed != "") base::as.integer(env_seed) else NULL
}
if (!base::is.null(seed)) {
  base::set.seed(seed)
  base::message("[R] set.seed(", seed, ")")
}

# Backend options:
#   mclapply  default; no future package required; Linux/HPC-friendly fork parallelism.
#   future    uses future/future.apply if explicitly requested.
#   serial    useful for debugging.
parallel_backend <- base::tolower(base::Sys.getenv("EVE_PARALLEL_BACKEND", unset = "mclapply"))
if (!parallel_backend %in% c("mclapply", "future", "serial")) {
  base::stop("EVE_PARALLEL_BACKEND must be one of: mclapply, future, serial")
}

if (base::requireNamespace("RcppParallel", quietly = TRUE)) {
  RcppParallel::setThreadOptions(numThreads = 1)
} else {
  base::message("Warning: RcppParallel is not visible; skipping RcppParallel::setThreadOptions(numThreads = 1).")
}

# Local BD simulator that exports in the same six-parameter EVE filename format:
# pars = lambda, mu, beta_n, beta_phi, gamma_n, gamma_phi.
randomized_bd_as_eve_fixed_age <- function(dists, age, min_ntip = 10, size_limit = 2000, max_attempts = 200) {
  for (attempt in base::seq_len(max_attempts)) {
    lambda <- eveGNN::generate_params(base::list(dists[[1]]))[[1]]
    mu <- stats::runif(1, min = 0, max = 0.8 * lambda)

    tas <- base::tryCatch({
      ape::rlineage(birth = lambda, death = mu, Tmax = age)
    }, error = function(e) NULL)
    if (base::is.null(tas)) next

    tes <- base::tryCatch({
      ape::drop.fossil(tas)
    }, error = function(e) NULL)
    if (base::is.null(tes)) next

    ntip <- ape::Ntip(tes)
    if (ntip >= min_ntip && ntip <= size_limit) {
      result <- base::list()
      result$tes <- tes
      result$pars <- c(lambda, mu, 0, 0, 0, 0)
      result$age <- age
      result$model <- "BD"
      result$metric <- "bd"
      result$offset <- "none"
      return(result)
    }
  }
  base::list(tes = NULL)
}

remove_nulls <- function(lst) {
  lst[base::vapply(lst, function(x) !base::is.null(x$tes), logical(1))]
}

transpose_sim_list <- function(lst, label) {
  lst <- remove_nulls(lst)
  if (base::length(lst) == 0) {
    base::stop("No non-null trees were generated for scenario: ", label)
  }
  keys <- base::unique(base::unlist(base::lapply(lst, base::names)))
  out <- base::lapply(keys, function(k) base::lapply(lst, function(x) x[[k]]))
  base::names(out) <- keys
  out
}

simulate_replicates <- function(label, nrep, generator, nworkers, backend) {
  base::message("[R] Simulating ", label, " trees with backend=", backend, ", workers=", nworkers, ", nrep=", nrep)

  if (nworkers <= 1 || backend == "serial") {
    return(base::lapply(base::seq_len(nrep), function(i) generator()))
  }

  if (backend == "mclapply") {
    require_namespace("parallel", install_hint = FALSE)
    return(parallel::mclapply(
      X = base::seq_len(nrep),
      FUN = function(i) generator(),
      mc.cores = nworkers,
      mc.set.seed = TRUE
    ))
  }

  if (backend == "future") {
    require_namespace("future")
    require_namespace("future.apply")
    future::plan(future::multicore, workers = nworkers)
    return(future.apply::future_lapply(base::seq_len(nrep), function(i) generator()))
  }

  base::stop("Unknown backend: ", backend)
}

if (parallel_backend == "future") {
  on.exit({
    if (base::requireNamespace("future", quietly = TRUE)) {
      future::plan(future::sequential)
    }
  }, add = TRUE)
}

base::message("[R] nrep=", nrep, "; age=", age, "; min_ntip=", min_ntip, "; size_limit=", size_limit)

eve_bd_list <- simulate_replicates(
  label = "BD baseline",
  nrep = nrep,
  generator = function() randomized_bd_as_eve_fixed_age(dists_bd, age = age, min_ntip = min_ntip, size_limit = size_limit),
  nworkers = nworkers_sim,
  backend = parallel_backend
)

eve_ed_list <- simulate_replicates(
  label = "ED",
  nrep = nrep,
  generator = function() eveGNN::randomized_eve_fixed_age(dists_ed, age = age, metric = "ed", offset = "none"),
  nworkers = nworkers_sim,
  backend = parallel_backend
)

eve_nnd_list <- simulate_replicates(
  label = "NND",
  nrep = nrep,
  generator = function() eveGNN::randomized_eve_fixed_age(dists_nnd, age = age, metric = "nnd", offset = "none"),
  nworkers = nworkers_sim,
  backend = parallel_backend
)

eve_bd_list_all <- transpose_sim_list(eve_bd_list, "BD")
eve_ed_list_all <- transpose_sim_list(eve_ed_list, "ED")
eve_nnd_list_all <- transpose_sim_list(eve_nnd_list, "NND")

if (!base::dir.exists(task_type)) {
  base::dir.create(task_type)
}
base::setwd(task_type)

base::message("[R] Exporting BD/ED/NND Training/Testing TES Data to GNN folders")
eveGNN::export_to_gnn_with_params_eve(eve_bd_list_all, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all, "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all, "tes", undirected = FALSE)

base::setwd("..")
if (!base::dir.exists("EVE_VAL_TES")) {
  base::dir.create("EVE_VAL_TES")
}

base::message("[R] Simulation/export complete.")
