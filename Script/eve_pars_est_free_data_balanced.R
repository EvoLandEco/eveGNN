args <- commandArgs(TRUE)

name <- as.character(args[1])

params <- yaml::read_yaml("../Config/eve_sim.yaml")

# Utilities
randomized_eve_fixed_age <- function(dists, age, metric, offset) {
  result <- list()
  params <- eveGNN::generate_params(dists)
  lambda <- params[[1]]
  mu <- runif(1, min = 0, max = 0.8 * lambda)
  beta_n <- params[[2]]; beta_phi <- params[[3]]
  gamma_n <- params[[4]]; gamma_phi <- params[[5]]

  pars_list <- c(lambda, mu, beta_n, beta_phi, gamma_n, gamma_phi)
  age <- as.double(age)
  raw_result <- evesim::edd_sim(pars = pars_list, age = age, metric = metric,
                                offset = offset, size_limit = 2000)

  if (is.null(raw_result$sim)) {
    result[["tes"]] <- NULL
    result[["n_nodes"]] <- NA_integer_
  } else {
    phy <- raw_result$sim
    ## total nodes = tips + internal nodes (ape 'phylo' convention)
    n_nodes <- length(phy$tip.label) + phy$Nnode   # ← add
    result[["n_nodes"]] <- n_nodes                 # ← add
    result[["tes"]] <- evesim::SimTable.phylo(phy, drop_extinct = TRUE)
  }

  result[["pars"]]   <- pars_list
  result[["age"]]    <- age
  result[["model"]]  <- "dsde2"
  result[["metric"]] <- metric
  result[["offset"]] <- offset
  return(result)
}

fill_size_quota <- function(dists, age, metric, offset,
                            sizes = 10:1000,         # or 10:1009 if need 100,000
                            per_size = 100,          # target per distinct size
                            batch_nrep = 5000,       # sims per batch (parallelized)
                            max_batches = 500,       # safety cap
                            nworkers_sim = parallel::detectCores() - 1) {

  target_total <- length(sizes) * per_size
  target       <- setNames(rep(per_size, length(sizes)), sizes)
  buckets      <- setNames(vector("list", length(sizes)), sizes)
  counts       <- setNames(integer(length(sizes)), sizes)

  future::plan("multicore", workers = nworkers_sim)
  RcppParallel::setThreadOptions(numThreads = 1)

  for (b in seq_len(max_batches)) {
    sims <- future.apply::future_lapply(seq_len(batch_nrep), function(i) {
      randomized_eve_fixed_age(dists, age = age, metric = metric, offset = offset)
    })

    # keep successful sims whose tree size is in target 'sizes'
    sims <- Filter(function(x) !is.null(x$tes) && is.finite(x$n_nodes) &&
      x$n_nodes %in% sizes, sims)

    # split by realized size, and add up to the remaining quota
    by_size <- split(sims, vapply(sims, function(s) as.character(s$n_nodes), ""))
    for (s in intersect(names(by_size), names(target))) {
      need <- target[[s]] - counts[[s]]
      if (need > 0L) {
        take <- utils::head(by_size[[s]], need)
        if (length(take)) {
          buckets[[s]] <- c(buckets[[s]], take)
          counts[[s]]  <- counts[[s]] + length(take)
        }
      }
    }

    if (all(counts >= target)) break
  }

  # sanity: which sizes didn’t reach quota?
  unmet <- names(counts)[counts < target]
  if (length(unmet)) {
    warning(sprintf("Quota not fully met for sizes: %s", paste(unmet, collapse = ", ")))
  }

  # flatten and transpose to the shape your exporter expects
  kept <- unlist(buckets, recursive = FALSE, use.names = FALSE)
  kept_nonnull <- Filter(function(x) !is.null(x$tes), kept)
  purrr::transpose(kept_nonnull)
}

if (!dir.exists(name)) {
  dir.create(name)
}

setwd(name)

dists_pd <- params$dists_pd
dists_ed <- params$dists_ed
dists_nnd <- params$dists_nnd

# Manually set betas and gammas to zero, for 2-Pars or 4-pars simulation
# These lines set betas to zero
dists_pd[[2]]$max <- 0
dists_pd[[2]]$min <- 0
dists_ed[[2]]$max <- 0
dists_ed[[2]]$min <- 0
dists_nnd[[2]]$max <- 0
dists_nnd[[2]]$min <- 0
dists_pd[[3]]$max <- 0
dists_pd[[3]]$min <- 0
dists_ed[[3]]$max <- 0
dists_ed[[3]]$min <- 0
dists_nnd[[3]]$max <- 0
dists_nnd[[3]]$min <- 0

# These lines set gammas to zero
# dists_pd[[4]]$max <- 0
# dists_pd[[4]]$min <- 0
# dists_ed[[4]]$max <- 0
# dists_ed[[4]]$min <- 0
# dists_nnd[[4]]$max <- 0
# dists_nnd[[4]]$min <- 0
# dists_pd[[5]]$max <- 0
# dists_pd[[5]]$min <- 0
# dists_ed[[5]]$max <- 0
# dists_ed[[5]]$min <- 0
# dists_nnd[[5]]$max <- 0
# dists_nnd[[5]]$min <- 0

nrep <- params$nrep
age <- params$age
nworkers_sim <- params$nworkers_sim

sizes      <- 10:1009
per_size   <- 50
age        <- params$age

# Build three balanced datasets
eve_pd_list_all  <- fill_size_quota(dists_pd,  age, metric = "pd",  offset = "simtime",
                                    sizes = sizes, per_size = per_size,
                                    batch_nrep = 5000, max_batches = 500, nworkers_sim = nworkers_sim)

eve_ed_list_all  <- fill_size_quota(dists_ed,  age, metric = "ed",  offset = "none",
                                    sizes = sizes, per_size = per_size,
                                    batch_nrep = 5000, max_batches = 500, nworkers_sim = nworkers_sim)

eve_nnd_list_all <- fill_size_quota(dists_nnd, age, metric = "nnd", offset = "none",
                                    sizes = sizes, per_size = per_size,
                                    batch_nrep = 5000, max_batches = 500, nworkers_sim = nworkers_sim)

# Export
dir.create("EVE_FREE_TES", showWarnings = FALSE)
setwd("EVE_FREE_TES")
message("Exporting Training/Testing TES Data to GNN")
eveGNN::export_to_gnn_with_params_eve(eve_pd_list_all,  "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_ed_list_all,  "tes", undirected = FALSE)
eveGNN::export_to_gnn_with_params_eve(eve_nnd_list_all, "tes", undirected = FALSE)

if (!dir.exists("EVE_VAL_TES")) {
  dir.create("EVE_VAL_TES")
}
