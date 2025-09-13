# ====================== spectral_phylo_LM_RFF_consolidated.R ======================
# Lewitus & Morlon (2016): spectral density profile (SDP) from Modified Graph Laplacian (MGL)
# + degree-normalized MGL (nMGL). We compute classic Gaussian-kernel SDPs (RPANDA-like)
# and bandwidth-robust Random Fourier Features (RFF) SDPs; export tidy data; compute JSD;
# and visualize profiles and unrolled time-series.
#
# Key sources:
# - Lewitus & Morlon (2016) "Characterizing and Comparing Phylogenies from their Laplacian Spectrum".
#   (definitions, SDP, eigengap).  [SysBio]  (spectral density via Gaussian kernel).  (MGL/nMGL).  :contentReference[oaicite:1]{index=1}
# - RPANDA::spectR / ::JSDtree (standard= MGL, normal = normalized variant).           :contentReference[oaicite:2]{index=2}
# - phytools::treeSlice for rootwards slicing (time-unrolled series).                   :contentReference[oaicite:3]{index=3}
#
# INPUT (expected by main driver analyze_forest / analyze_with_rff):
#   a tibble/data.frame with columns:
#     tree_id (unique id), metric ∈ {pd, ed, nnd}, tree (phylo),
#     lambda, mu, beta_n, beta_phi, gamma_n, gamma_phi (optional but recommended)
#
# OUTPUT (written to OUT_DIR):
#   - spectra_full_long.csv           : full-tree Gaussian SDPs (tidy)
#   - spectra_summary.csv             : per-tree scalar stats (RPANDA + band masses + size & params)
#   - spectra_unrolled_long.csv       : unrolled (height × λ) Gaussian SDPs (tidy, nMGL)
#   - rff_profiles_full_long.csv      : full-tree RFF SDPs across bandwidths (tidy, MGL & nMGL)
#   - rff_profiles_unrolled_long.csv  : unrolled RFF SDPs (tidy, nMGL)
#   - JSD_RPANDA_standard_*.csv / JSD_RPANDA_normal_*.csv  (per metric)
#   - JSD_RFF_nMGL_h=*.csv            : per-bandwidth RFF JSD (tree × tree)
#
# (Optional) You can integrate simulation via your function:
# randomized_eve_fixed_age(dists, age, metric, offset) -> list(tes=phylo, pars=c(lambda,mu,beta_n,beta_phi,gamma_n,gamma_phi), ...)

suppressPackageStartupMessages({
  library(RPANDA)     # spectR, JSDtree
  library(ape)
  library(phytools)   # treeSlice, nodeHeights
  library(phangorn)   # dist.nodes
  library(Matrix)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(ggplot2)
  library(ggridges)
  library(parallel)
})

# --------------------------- CONFIG ---------------------------------------------
SPEC_GRID_N  <- 512                 # grid points for Gaussian SDPs
BAND_SPLITS  <- c(0.2, 1.2)         # low/mid/high boundaries (nMGL λ)
OUT_DIR <- "../spectral_out"; dir.create(OUT_DIR, showWarnings = FALSE)

# RFF defaults
RFF_BANDWIDTHS <- c(0.03, 0.05, 0.1, 0.2, 0.4)  # kernel bandwidths h to sweep
RFF_M_FULL     <- 2048
RFF_REPS_FULL  <- 3
RFF_M_UNROLLED <- 1024
RFF_REPS_UNROLLED <- 2
RFF_SEED       <- 123

# --------------------------- Utilities ------------------------------------------
`%||%` <- function(a,b) if (!is.null(a)) a else b

trapz <- function(x, y) {
  if (length(x) < 2) return(0)
  sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
}

band_masses <- function(x, y, splits = BAND_SPLITS) {
  x <- as.numeric(x); y <- as.numeric(y)
  if (!isTRUE(all(diff(x) > 0))) {
    o <- order(x); x <- x[o]; y <- y[o]
  }
  s1 <- splits[1]; s2 <- splits[2]
  i1 <- findInterval(s1, x); i2 <- findInterval(s2, x)
  low  <- if (i1 >= 1) trapz(x[1:i1], y[1:i1]) else 0
  mid  <- if (i2 > i1) trapz(x[(i1+1):i2], y[(i1+1):i2]) else 0
  high <- if (length(x) > i2) trapz(x[(i2+1):length(x)], y[(i2+1):length(x)]) else 0
  c(low = low, mid = mid, high = high)
}

# ---------------------- MGL / nMGL eigenvalues (all nodes) -----------------------
patristic_all_nodes <- function(tree) {
  W <- try(as.matrix(phangorn::dist.nodes(tree)), silent = TRUE)
  if (inherits(W, "try-error")) W <- as.matrix(ape::dist.nodes(tree))
  W <- 0.5 * (W + t(W)); diag(W) <- 0
  W
}

mgl_nmgl_eigen <- function(tree) {
  W <- patristic_all_nodes(tree)
  d <- rowSums(W); D <- diag(d)
  L  <- D - W
  invsqrt <- 1 / sqrt(pmax(d, 1e-12))
  Dm12 <- diag(invsqrt)
  Ln <- Dm12 %*% L %*% Dm12
  ev_MGL  <- sort(Re(eigen(L,  symmetric = TRUE, only.values = TRUE)$values))
  ev_nMGL <- sort(Re(eigen(Ln, symmetric = TRUE, only.values = TRUE)$values))
  list(MGL = ev_MGL, nMGL = ev_nMGL)
}

# ---------------------- Gaussian SDP (RPANDA-like) ------------------------------
gaussian_sdp <- function(evals, grid = NULL, bw = NULL, normalize = TRUE) {
  if (is.null(grid)) {
    r <- range(evals); pad <- 0.05 * max(1, diff(r))
    grid <- seq(r[1] - pad, r[2] + pad, length.out = SPEC_GRID_N)
  }
  if (is.null(bw)) {
    n <- length(evals); s <- stats::sd(evals); bw <- 0.9 * min(s, IQR(evals)/1.34) * n^(-1/5)
    if (!is.finite(bw) || bw <= 0) bw <- 0.05 * max(1, diff(range(evals)))
  }
  y <- vapply(grid, function(x) mean(stats::dnorm(x, mean = evals, sd = bw)), numeric(1))
  if (normalize) {
    dx <- mean(diff(grid)); s <- sum(y) * dx; if (s > 0) y <- y / s
  }
  list(x = grid, y = y, bw = bw)
}

# ---------------------- RPANDA stats wrappers -----------------------------------
rpanda_stats <- function(tree, method = c("standard","normal")) {
  method <- match.arg(method)
  s <- try(RPANDA::spectR(tree, method = method, zero_bound = FALSE), silent = TRUE)
  if (inherits(s, "try-error")) s <- try(RPANDA::spectR(tree, meth = method, zero_bound = FALSE), silent = TRUE)
  if (inherits(s, "try-error")) {
    return(list(principal_eigenvalue = NA_real_, eigengap = NA_real_,
                asymmetry = NA_real_, peakedness = NA_real_))
  }
  list(
    principal_eigenvalue = s$principal_eigenvalue %||% NA_real_,
    eigengap             = s$eigengap %||% NA_real_,
    asymmetry            = s$asymmetry %||% NA_real_,
    peakedness           = (s$peak_height %||% s$peakedness2 %||% NA_real_)
  )
}

jsd_rpanda_matrix <- function(trees, method = c("standard","normal1", "normal2")) {
  method <- match.arg(method)
  M <- RPANDA::JSDtree(trees, meth = method)
  M
}

# ---------------------- RFF SDP (bandwidth-swept) -------------------------------
# Random Fourier Features for Gaussian kernel density over eigenvalues
# k_h(x,y) = (1/(sqrt(2*pi)h)) * exp(-(x-y)^2/(2h^2))  ≈  (1/(sqrt(2*pi)h)) E_{ω~N(0,h^{-2}),b~U}[cos(ωx+b)cos(ωy+b)]
rff_sdp_1d <- function(evals, grid, bandwidths, m = 2048, reps = 1, seed = 42,
                       normalize = TRUE, clamp_nonneg = TRUE) {
  stopifnot(is.numeric(evals), is.numeric(grid), is.numeric(bandwidths))
  set.seed(seed)
  xi <- rnorm(m)               # base standard-normal freqs to be scaled by 1/h
  base_b <- runif(m, 0, 2*pi)  # base phases (will refresh per replicate)
  out <- vector("list", length(bandwidths) * reps)
  k <- 0L
  for (r in seq_len(reps)) {
    b <- if (r == 1) base_b else runif(m, 0, 2*pi)
    for (h in bandwidths) {
      omega <- xi / h
      ME <- outer(evals, omega)                    # n_eval × m
      PhiE <- sqrt(2/m) * cos(ME + rep(b, each = nrow(ME)))
      mu_hat <- colMeans(PhiE)                     # m
      MG <- outer(grid, omega)                     # n_grid × m
      PhiG <- sqrt(2/m) * cos(MG + rep(b, each = nrow(MG)))
      dens <- as.numeric(PhiG %*% mu_hat) * (1/(sqrt(2*pi)*h))
      if (clamp_nonneg) dens <- pmax(dens, 0)
      if (normalize) {
        dx <- mean(diff(grid)); s <- sum(dens) * dx; if (s > 0) dens <- dens / s
      }
      k <- k + 1L
      out[[k]] <- data.frame(h = h, rep = r, lambda = grid, density = dens)
    }
  }
  do.call(rbind, out)
}

# Discrete Jensen–Shannon divergence between two densities on same grid
js_div <- function(p, q, eps = 1e-12) {
  p <- p / (sum(p) + eps); q <- q / (sum(q) + eps); m <- 0.5*(p+q)
  kl <- function(a,b) sum(a * (log(a+eps)-log(b+eps)))
  (0.5*kl(p,m) + 0.5*kl(q,m)) / log(2)  # ∈ [0,1]
}

# ---------------------- One-tree analysis (Gaussian + RFF) ----------------------
# Turn a df with columns (tes, tas) into long format with (tree, view)
explode_views <- function(df, which = c("both","tes","tas")) {
  which <- match.arg(which)
  has_tes <- "tes" %in% names(df); has_tas <- "tas" %in% names(df)
  if (!has_tes && !has_tas && "tree" %in% names(df)) {
    df$view <- "tes"; return(dplyr::rename(df, tree = .data$tree))
  }
  rows <- list()
  if (which %in% c("both","tes") && has_tes) {
    rows$tes <- df %>% dplyr::mutate(tree = .data$tes, view = "tes") %>%
      dplyr::select(-"tes")
  }
  if (which %in% c("both","tas") && has_tas) {
    rows$tas <- df %>% dplyr::mutate(tree = .data$tas, view = "tas") %>%
      dplyr::select(-"tas")
  }
  dplyr::bind_rows(rows)
}

# ---- One-tree analysis (Gaussian + RFF) ----
analyze_tree <- function(tree, tree_id, metric, params = NULL,
                         grid_MGL = NULL, grid_nMGL = seq(0,2,length.out=SPEC_GRID_N),
                         norm_prob = TRUE,
                         rff_bandwidths = RFF_BANDWIDTHS,
                         rff_m = RFF_M_FULL, rff_reps = RFF_REPS_FULL, rff_seed = RFF_SEED,
                         view = NA_character_) {
  ev <- mgl_nmgl_eigen(tree)
  if (is.null(grid_MGL)) {
    rg <- range(ev$MGL); pad <- 0.05 * max(1, diff(rg))
    grid_MGL <- seq(rg[1]-pad, rg[2]+pad, length.out = SPEC_GRID_N)
  }
  dens_MGL  <- gaussian_sdp(ev$MGL,  grid = grid_MGL,  normalize = norm_prob)
  dens_nMGL <- gaussian_sdp(ev$nMGL, grid = grid_nMGL, normalize = norm_prob)

  rp_std <- rpanda_stats(tree, "standard")
  rp_nrm <- rpanda_stats(tree, "normal")

  df_full <- dplyr::bind_rows(
    tibble::tibble(tree_id, metric, view, lap="MGL",  lambda=dens_MGL$x,  density=dens_MGL$y),
    tibble::tibble(tree_id, metric, view, lap="nMGL", lambda=dens_nMGL$x, density=dens_nMGL$y)
  )
  bands <- band_masses(dens_nMGL$x, dens_nMGL$y, BAND_SPLITS)

  df_sum <- tibble::tibble(
    tree_id, metric, view,
    n_tips = Ntip(tree), n_nodes = tree$Nnode + Ntip(tree),
    lambda = params["lambda"] %||% NA_real_,
    mu     = params["mu"]     %||% NA_real_,
    beta_n = params["beta_n"] %||% NA_real_,
    beta_phi = params["beta_phi"] %||% NA_real_,
    gamma_n  = params["gamma_n"]  %||% NA_real_,
    gamma_phi= params["gamma_phi"]%||% NA_real_,
    rp_std_princ = rp_std$principal_eigenvalue, rp_std_gap = rp_std$eigengap,
    rp_std_asym  = rp_std$asymmetry,           rp_std_peak= rp_std$peakedness,
    rp_nrm_princ = rp_nrm$principal_eigenvalue, rp_nrm_gap = rp_nrm$eigengap,
    rp_nrm_asym  = rp_nrm$asymmetry,            rp_nrm_peak= rp_nrm$peakedness,
    nMGL_band_low = bands["low"], nMGL_band_mid = bands["mid"], nMGL_band_high = bands["high"]
  )

  rff_nMGL <- rff_sdp_1d(ev$nMGL, grid = grid_nMGL, bandwidths = rff_bandwidths,
                         m = rff_m, reps = rff_reps, seed = rff_seed, normalize = TRUE)
  rff_nMGL$lap <- "nMGL"; rff_nMGL$tree_id <- tree_id; rff_nMGL$metric <- metric; rff_nMGL$view <- view
  rff_MGL  <- rff_sdp_1d(ev$MGL,  grid = grid_MGL,  bandwidths = rff_bandwidths,
                         m = rff_m, reps = rff_reps, seed = rff_seed, normalize = TRUE)
  rff_MGL$lap <- "MGL";  rff_MGL$tree_id <- tree_id;  rff_MGL$metric <- metric;  rff_MGL$view <- view
  df_full_rff <- dplyr::bind_rows(rff_nMGL, rff_MGL)

  list(df_full = df_full, df_sum = df_sum, df_full_rff = df_full_rff)
}

# ---- helpers ----
.ensure_rooted <- function(tr) {
  if (!ape::is.rooted(tr)) {
    tr <- phangorn::midpoint(tr)
  }
  tr
}

.clean_heights <- function(tree, skip_first = TRUE, eps_rel = 1e-8) {
  H <- phytools::nodeHeights(tree)              # E x 2
  child_ids <- tree$edge[,2]
  r_int <- which(child_ids > Ntip(tree))
  h <- sort(unique(H[r_int, 2]))
  h <- h[is.finite(h)]
  maxH <- max(H)
  eps  <- max(maxH, 1) * eps_rel
  # keep strictly between (0, maxH)
  h <- h[(h > eps) & (h < (maxH - eps))]
  if (skip_first && length(h) > 1) h <- h[-1]   # drop the earliest (first) internal split
  h
}

# ---- robust Gaussian unrolling (nMGL) ----
unroll_tree_series <- function(tree, tree_id, metric, view,
                               grid_nMGL = seq(0,2,length.out=256),
                               norm_prob = TRUE,
                               min_nodes = 3, min_tips = 2,
                               skip_first = TRUE) {
  tree <- .ensure_rooted(tree)
  heights <- .clean_heights(tree, skip_first = skip_first)
  out <- vector("list", length(heights)); k <- 0L
  for (h in heights) {
    trh <- try(phytools::treeSlice(tree, slice = h, trivial = FALSE,
                                   prompt = FALSE, orientation = "rootwards"),
               silent = TRUE)
    if (inherits(trh, "try-error") || is.null(trh)) next
    if (inherits(trh, "multiPhylo")) {
      sz <- sapply(trh, Ntip); trh <- trh[[which.max(sz)]]
    }
    if ((Ntip(trh) < min_tips) || (trh$Nnode + Ntip(trh) < min_nodes)) next
    ev <- mgl_nmgl_eigen(trh)
    dn <- gaussian_sdp(ev$nMGL, grid = grid_nMGL, normalize = norm_prob)
    k <- k + 1L
    out[[k]] <- tibble::tibble(tree_id, metric, view = view, height = h,
                               lambda = dn$x, density = dn$y, lap = "nMGL")
  }
  if (k == 0L) return(tibble::tibble(tree_id, metric, height = numeric(0),
                                     lambda = numeric(0), density = numeric(0), lap = "nMGL"))
  dplyr::bind_rows(out[seq_len(k)])
}

# ---- robust RFF unrolling (nMGL) ----
rff_unrolled_for_tree <- function(tree, tree_id, metric, view,
                                  bandwidths = RFF_BANDWIDTHS,
                                  grid_nMGL  = seq(0, 2, length.out = 256),
                                  m = RFF_M_UNROLLED, reps = RFF_REPS_UNROLLED, seed = RFF_SEED,
                                  min_nodes = 3, min_tips = 2,
                                  skip_first = TRUE) {
  tree <- .ensure_rooted(tree)
  heights <- .clean_heights(tree, skip_first = skip_first)
  out <- vector("list", length(heights)); k <- 0L
  for (h in heights) {
    trh <- try(phytools::treeSlice(tree, slice = h, trivial = FALSE,
                                   prompt = FALSE, orientation = "rootwards"),
               silent = TRUE)
    if (inherits(trh, "try-error") || is.null(trh)) next
    if (inherits(trh, "multiPhylo")) {
      sz <- sapply(trh, Ntip); trh <- trh[[which.max(sz)]]
    }
    if ((Ntip(trh) < min_tips) || (trh$Nnode + Ntip(trh) < min_nodes)) next
    ev <- mgl_nmgl_eigen(trh)
    df <- rff_sdp_1d(evals = ev$nMGL, grid = grid_nMGL, bandwidths = bandwidths,
                     m = m, reps = reps, seed = seed, normalize = TRUE)
    df$tree_id <- tree_id
    df$metric <- metric
    df$view <- view
    df$height <- h
    df$lap <- "nMGL"
    k <- k + 1L
    out[[k]] <- df
  }
  if (k == 0L) return(tibble::tibble(tree_id=tree_id, metric=metric, height=numeric(0)))
  dplyr::bind_rows(out[seq_len(k)])
}

# --- total height (optionally include stem/root.edge) & rescale to target crown age ---
.tree_total_height <- function(tree, include_root_edge = TRUE) {
  H <- phytools::nodeHeights(tree)
  h <- max(H)
  if (include_root_edge && !is.null(tree$root.edge)) h <- h + tree$root.edge
  h
}

.rescale_tree_to_age <- function(tree, target_age, include_root_edge = TRUE) {
  cur <- .tree_total_height(tree, include_root_edge = include_root_edge)
  if (!is.finite(cur) || cur <= 0) return(tree)
  s <- target_age / cur
  tree$edge.length <- tree$edge.length * s
  if (include_root_edge && !is.null(tree$root.edge)) tree$root.edge <- tree$root.edge * s
  tree
}

# ---- Uniform-time Gaussian unrolling (nMGL) ----
unroll_tree_series_timegrid <- function(
  tree, tree_id, metric, view,
  age_target = 10.0,        # desired common crown age
  k = 20L,                  # number of time points (including both ends)
  rescale_each_tree = TRUE, # rescale to age_target before slicing
  include_root_edge = TRUE,
  grid_nMGL = seq(0, 2, length.out = 256),
  norm_prob = TRUE,
  min_nodes = 3, min_tips = 2
) {
  tree <- .ensure_rooted(tree)
  if (isTRUE(rescale_each_tree)) tree <- .rescale_tree_to_age(tree, age_target, include_root_edge)

  # Even grid [0, age_target]; avoid exact endpoints for slicing (numerical/logic issues)
  tt <- seq(0, age_target, length.out = k)
  eps <- max(1e-8 * age_target, .Machine$double.eps)
  slices <- tt[tt > eps & tt < (age_target - eps)]

  out <- vector("list", length(slices)); j <- 0L
  for (h in slices) {
    trh <- try(phytools::treeSlice(tree, slice = h, trivial = FALSE,
                                   prompt = FALSE, orientation = "rootwards"),
               silent = TRUE)
    if (inherits(trh, "try-error") || is.null(trh)) next
    if (inherits(trh, "multiPhylo")) { sz <- sapply(trh, Ntip); trh <- trh[[which.max(sz)]] }
    if ((Ntip(trh) < min_tips) || (trh$Nnode + Ntip(trh) < min_nodes)) next

    ev <- mgl_nmgl_eigen(trh)
    dn <- gaussian_sdp(ev$nMGL, grid = grid_nMGL, normalize = norm_prob)
    j <- j + 1L
    out[[j]] <- tibble::tibble(
      tree_id, metric,
      view = view,
      height = h,                         # absolute time since root (after rescaling)
      t_norm = h / age_target,            # normalized time in [0,1]
      lambda = dn$x, density = dn$y, lap = "nMGL"
    )
  }
  if (j == 0L) return(tibble::tibble(tree_id, metric, height = numeric(0), t_norm = numeric(0),
                                     lambda = numeric(0), density = numeric(0), lap = "nMGL"))
  dplyr::bind_rows(out[seq_len(j)])
}

# ---- Uniform-time RFF unrolling (nMGL) ----
rff_unrolled_for_tree_timegrid <- function(
  tree, tree_id, metric, view,
  age_target = 10.0, k = 20L,
  rescale_each_tree = TRUE, include_root_edge = TRUE,
  bandwidths = RFF_BANDWIDTHS,
  grid_nMGL  = seq(0, 2, length.out = 256),
  m = RFF_M_UNROLLED, reps = RFF_REPS_UNROLLED, seed = RFF_SEED,
  min_nodes = 3, min_tips = 2
) {
  tree <- .ensure_rooted(tree)
  if (isTRUE(rescale_each_tree)) tree <- .rescale_tree_to_age(tree, age_target, include_root_edge)
  tt <- seq(0, age_target, length.out = k)
  eps <- max(1e-8 * age_target, .Machine$double.eps)
  slices <- tt[tt > eps & tt < (age_target - eps)]

  out <- vector("list", length(slices)); j <- 0L
  for (h in slices) {
    trh <- try(phytools::treeSlice(tree, slice = h, trivial = FALSE,
                                   prompt = FALSE, orientation = "rootwards"),
               silent = TRUE)
    if (inherits(trh, "try-error") || is.null(trh)) next
    if (inherits(trh, "multiPhylo")) { sz <- sapply(trh, Ntip); trh <- trh[[which.max(sz)]] }
    if ((Ntip(trh) < min_tips) || (trh$Nnode + Ntip(trh) < min_nodes)) next

    ev <- mgl_nmgl_eigen(trh)
    df <- rff_sdp_1d(evals = ev$nMGL, grid = grid_nMGL, bandwidths = bandwidths,
                     m = m, reps = reps, seed = seed, normalize = TRUE)
    df$tree_id <- tree_id
    df$metric <- metric
    df$view <- view
    df$height <- h
    df$t_norm <- h / age_target
    df$lap <- "nMGL"
    j <- j + 1L; out[[j]] <- df
  }
  if (j == 0L) return(tibble::tibble(tree_id=tree_id, metric=metric, height=numeric(0), t_norm=numeric(0)))
  dplyr::bind_rows(out[seq_len(j)])
}

# ---------------------- End-to-end drivers --------------------------------------
# df must include: tree_id, metric, tree, and (optionally) true params
analyze_forest <- function(
  df,
  parallel = TRUE,
  ncores   = max(1L, (parallel::detectCores(logical = FALSE) %||% 2L) - 1L),
  backend  = c("auto","multicore","psock"),
  rng_seed = 123,
  unroll_mode = c("events","grid"),
  grid_k = 20L,
  grid_age_target = 10.0,
  grid_rescale_each_tree = TRUE,
  which_view = c("both","tes","tas")
) {
  which_view <- match.arg(which_view)
  backend <- match.arg(backend)
  if (!parallel || ncores <= 1L) backend <- "serial"
  if (backend == "auto") backend <- if (.Platform$OS.type == "windows") "psock" else "multicore"

  df <- explode_views(df, which = which_view)
  stopifnot(all(c("tree","view") %in% names(df)))

  # -------- helper closures over 'df' --------
  # one full-tree analysis (Gaussian + RFF)
  .do_analyze_one <- function(i) {
    tr <- df$tree[[i]]
    analyze_tree(
      tree    = tr,
      tree_id = df$tree_id[[i]],
      metric  = df$metric[[i]],
      params  = c(
        lambda    = df$lambda[[i]] %||% NA_real_,
        mu        = df$mu[[i]]     %||% NA_real_,
        beta_n    = df$beta_n[[i]] %||% NA_real_,
        beta_phi  = df$beta_phi[[i]] %||% NA_real_,
        gamma_n   = df$gamma_n[[i]] %||% NA_real_,
        gamma_phi = df$gamma_phi[[i]] %||% NA_real_
      ),
      view = df$view[[i]]
    )
  }
  .do_unroll_gauss <- function(i) {
    if (unroll_mode == "grid") {
      unroll_tree_series_timegrid(
        tree = df$tree[[i]], tree_id = df$tree_id[[i]], metric = df$metric[[i]],
        view = df$view[[i]],
        age_target = grid_age_target, k = grid_k, rescale_each_tree = grid_rescale_each_tree
      )
    } else {
      unroll_tree_series(tree = df$tree[[i]], tree_id = df$tree_id[[i]], metric = df$metric[[i]], view = df$view[[i]])
    }
  }

  .do_unroll_rff <- function(i) {
    if (unroll_mode == "grid") {
      rff_unrolled_for_tree_timegrid(
        tree = df$tree[[i]], tree_id = df$tree_id[[i]], metric = df$metric[[i]],
        view = df$view[[i]],
        age_target = grid_age_target, k = grid_k, rescale_each_tree = grid_rescale_each_tree
      )
    } else {
      rff_unrolled_for_tree(tree = df$tree[[i]], tree_id = df$tree_id[[i]], metric = df$metric[[i]], view = df$view[[i]])
    }
  }

  # -------- run workers (serial or parallel) --------
  N <- nrow(df)
  parts_analyze <- parts_unroll_gauss <- parts_unroll_rff <- NULL

  if (backend == "serial") {
    # serial fall-back
    parts_analyze      <- lapply(seq_len(N), .do_analyze_one)
    parts_unroll_gauss <- lapply(seq_len(N), .do_unroll_gauss)
    parts_unroll_rff   <- lapply(seq_len(N), .do_unroll_rff)
  } else if (backend == "multicore") {
    # Unix/macOS forking
    set.seed(rng_seed)
    parts_analyze      <- parallel::mclapply(seq_len(N), .do_analyze_one,      mc.cores = ncores, mc.preschedule = TRUE)
    parts_unroll_gauss <- parallel::mclapply(seq_len(N), .do_unroll_gauss,     mc.cores = ncores, mc.preschedule = TRUE)
    parts_unroll_rff   <- parallel::mclapply(seq_len(N), .do_unroll_rff,       mc.cores = ncores, mc.preschedule = TRUE)
  } else {
    # PSOCK cluster (Windows or forced)
    ncores <- max(1L, ncores)
    cl <- parallel::makeCluster(ncores)
    on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)

    # Ensure libs on workers
    parallel::clusterEvalQ(cl, {
      suppressPackageStartupMessages({
        library(RPANDA); library(ape); library(phytools); library(phangorn); library(Matrix)
        library(data.table); library(dplyr); library(tidyr); library(tibble)
      })
      NULL
    })
    # Reproducible streams
    parallel::clusterSetRNGStream(cl, rng_seed)

    # Export data + all functions/constants referenced by workers
    parallel::clusterExport(
      cl,
      varlist = c(
        "df",
        # worker closures (PSOCK can export closures directly in recent R)
        ".do_analyze_one",".do_unroll_gauss",".do_unroll_rff",
        # called functions:
        "analyze_tree","unroll_tree_series","rff_unrolled_for_tree",
        "mgl_nmgl_eigen","gaussian_sdp","rpanda_stats","rff_sdp_1d",
        "js_div","BAND_SPLITS","SPEC_GRID_N",
        "RFF_BANDWIDTHS","RFF_M_FULL","RFF_REPS_FULL",
        "RFF_M_UNROLLED","RFF_REPS_UNROLLED","RFF_SEED"
      ),
      envir = environment()
    )

    parts_analyze      <- parallel::parLapply(cl, seq_len(N), function(i) .do_analyze_one(i))
    parts_unroll_gauss <- parallel::parLapply(cl, seq_len(N), function(i) .do_unroll_gauss(i))
    parts_unroll_rff   <- parallel::parLapply(cl, seq_len(N), function(i) .do_unroll_rff(i))
  }

  # -------- bind full-tree outputs --------
  df_full_gauss <- dplyr::bind_rows(lapply(parts_analyze, `[[`, "df_full"))
  df_sum        <- dplyr::bind_rows(lapply(parts_analyze, `[[`, "df_sum"))
  df_full_rff   <- dplyr::bind_rows(lapply(parts_analyze, `[[`, "df_full_rff"))

  # -------- bind unrolled outputs --------
  parts_unroll_gauss <- parts_unroll_gauss[!vapply(parts_unroll_gauss, is.null, logical(1))]
  parts_unroll_rff   <- parts_unroll_rff[!vapply(parts_unroll_rff,   is.null, logical(1))]
  df_series_gauss <- if (length(parts_unroll_gauss)) dplyr::bind_rows(parts_unroll_gauss) else
    tibble::tibble(tree_id=character(), metric=character(), view=character(), height=numeric(), lambda=numeric(), density=numeric(), lap=character())
  df_series_rff   <- if (length(parts_unroll_rff))   dplyr::bind_rows(parts_unroll_rff)   else
    tibble::tibble(tree_id=character(), metric=character(), view=character(), height=numeric(), h=numeric(), rep=integer(), lambda=numeric(), density=numeric(), lap=character())

  # -------- JSD by metric using RPANDA SDPs (baseline, keep serial) --------
  jsd_rpanda <- list()
  for (mv in work_df %>% dplyr::distinct(metric, view) %>% dplyr::arrange(metric, view) %>% split(.$metric)) {
    m <- unique(mv$metric)
    for (v in unique(work_df$view)) {
      trees <- work_df %>% dplyr::filter(metric == m, view == v) %>% dplyr::pull(tree)
      if (!length(trees)) next
      jsd_rpanda[[m]][[v]] <- list(
        standard = jsd_rpanda_matrix(trees, "standard"),
        normal   = jsd_rpanda_matrix(trees, "normal1")
      )
      data.table::fwrite(as.data.frame(jsd_rpanda[[m]][[v]]$standard),
        file.path(OUT_DIR, paste0("JSD_RPANDA_standard_", m, "_", v, ".csv")))
      data.table::fwrite(as.data.frame(jsd_rpanda[[m]][[v]]$normal),
        file.path(OUT_DIR, paste0("JSD_RPANDA_normal_", m, "_", v, ".csv")))
    }
  }

  # -------- RFF JSD per bandwidth (nMGL) --------
  jsd_rff_by_h <- list()
  tol <- 1e-9
  for (v in unique(df_full_rff$view)) {
    jsd_rff_by_h[[v]] <- list()
    for (h in RFF_BANDWIDTHS) {
      sub <- df_full_rff %>%
        dplyr::filter(view == v, lap == "nMGL", abs(.data$h - h) < tol) %>%
        dplyr::mutate(lambda = round(lambda, 8)) %>%
        dplyr::group_by(tree_id, lambda) %>%
        dplyr::summarise(density = mean(as.numeric(density)), .groups = "drop") %>%
        dplyr::arrange(tree_id, lambda)
      if (nrow(sub) == 0) next
      mats <- tidyr::pivot_wider(sub, names_from = lambda, values_from = density, values_fill = 0)
      mat <- data.matrix(mats[,-1, drop=FALSE]); rownames(mat) <- mats$tree_id
      n <- nrow(mat); M <- matrix(0, n, n, dimnames = list(rownames(mat), rownames(mat)))
      for (i in seq_len(n)) for (j in i:n) { d <- js_div(mat[i,], mat[j,]); M[i,j] <- d; M[j,i] <- d }
      jsd_rff_by_h[[v]][[as.character(h)]] <- M
      data.table::fwrite(as.data.frame(M), file.path(OUT_DIR, sprintf("JSD_RFF_nMGL_%s_h=%.3f.csv", v, h)))
    }
  }

  # -------- Save tidy CSVs --------
  data.table::fwrite(df_full_gauss,   file.path(OUT_DIR, "spectra_full_long.csv"))
  data.table::fwrite(df_sum,          file.path(OUT_DIR, "spectra_summary.csv"))
  data.table::fwrite(df_series_gauss, file.path(OUT_DIR, "spectra_unrolled_long.csv"))
  data.table::fwrite(df_full_rff,     file.path(OUT_DIR, "rff_profiles_full_long.csv"))
  data.table::fwrite(df_series_rff,   file.path(OUT_DIR, "rff_profiles_unrolled_long.csv"))

  # -------- Return --------
  list(
    full_gauss   = df_full_gauss,
    summary      = df_sum,
    series_gauss = df_series_gauss,
    full_rff     = df_full_rff,
    series_rff   = df_series_rff,
    jsd_rpanda   = jsd_rpanda,
    jsd_rff      = jsd_rff_by_h
  )
}

# ---------------------- Visualization helpers -----------------------------------
plot_profiles_gauss <- function(df_full, focus_lap = "nMGL") {
  df_full %>% filter(lap == focus_lap) %>%
    ggplot(aes(x = lambda, y = density, color = metric, group = interaction(tree_id, metric))) +
    geom_line(alpha = 0.35) +
    stat_summary(aes(group = metric), fun = mean, geom = "line", linewidth = 1.2) +
    theme_minimal() + labs(title = paste("Gaussian SDP (", focus_lap, ")", sep=""),
                           x = expression(lambda), y = "density")
}

plot_profiles_rff_by_h <- function(df_full_rff, focus_lap = "nMGL") {
  df_full_rff %>% filter(lap == focus_lap) %>%
    ggplot(aes(x = lambda, y = density, group = interaction(tree_id, rep), color = factor(h))) +
    geom_line(alpha = 0.25) +
    stat_summary(aes(group = factor(h), color = factor(h)), fun = mean, geom = "line", linewidth = 1.1) +
    facet_wrap(~ metric, scales = "free_y") +
    scale_color_brewer(palette="Dark2", name="bandwidth h") +
    theme_minimal() + labs(title=paste("RFF SDPs (",focus_lap,") by bandwidth h"),
                           x=expression(lambda), y="density")
}

plot_unrolled_heatmap_gauss <- function(df_series) {
  dfm <- df_series %>% group_by(metric, height, lambda) %>% summarise(density = mean(density), .groups="drop")
  ggplot(dfm, aes(x = height, y = lambda, fill = density)) +
    geom_raster() + facet_wrap(~ metric, scales="free_x") +
    scale_fill_viridis_c() + theme_minimal() +
    labs(title = "Time-unrolled Gaussian SDP (nMGL): mean across trees",
         x = "Height above root (time)", y = expression(lambda))
}

plot_unrolled_heatmap_rff <- function(df_series_rff) {
  dfm <- df_series_rff %>% group_by(metric, h, height, lambda) %>% summarise(density = mean(density), .groups="drop")
  ggplot(dfm, aes(x = height, y = lambda, fill = density)) +
    geom_raster() + facet_grid(metric ~ h, scales="free_x") +
    scale_fill_viridis_c() + theme_minimal() +
    labs(title = "Time-unrolled RFF SDP (nMGL): mean across trees",
         x = "Height above root (time)", y = expression(lambda))
}

plot_unrolled_ridgelines_rff <- function(df_series_rff, nbins = 8) {
  brks <- df_series_rff %>% distinct(height) %>% arrange(height) %>%
    pull(height) %>% quantile(probs = seq(0,1,length.out = nbins+1), na.rm = TRUE) %>% unique()
  dfr <- df_series_rff %>%
    mutate(hbin = cut(height, breaks=brks, include.lowest=TRUE, right=FALSE)) %>%
    group_by(metric, h, hbin, lambda) %>% summarise(density = mean(density), .groups="drop")
  ggplot(dfr, aes(x = lambda, y = hbin, height = density, fill = factor(h))) +
    ggridges::geom_ridgeline_gradient(scale=3, rel_min_height=0.001, alpha=0.85) +
    facet_wrap(~ metric, ncol = 1) + theme_minimal() +
    scale_fill_brewer(palette="Dark2", name="h") +
    labs(title="Ridgelines: unrolled RFF SDPs (nMGL)", y="time bins", x=expression(lambda))
}

plot_jsd_heatmap <- function(M, title = "JSD between profiles") {
  dn <- as.data.frame(M); dn$tree_i <- rownames(M)
  dlong <- dn %>% pivot_longer(-tree_i, names_to = "tree_j", values_to = "JSD")
  ggplot(dlong, aes(x = tree_i, y = tree_j, fill = JSD)) +
    geom_tile() + coord_equal() + scale_fill_viridis_c() +
    theme_minimal() + theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1)) +
    labs(title = title, x = "Tree", y = "Tree")
}

# ---------------------- Simulation wrappers --------------------------
# 10-point (inclusive) grid for a uniform spec
.mk_seq <- function(spec, n = 10L) {
  seq(from = as.numeric(spec$min), to = as.numeric(spec$max), length.out = n)
}

# Parallel simulator: one metric ("pd" | "ed" | "nnd")
simulate_metric_grid_parallel <- function(
  dists,                 # 5-element list: lambda, beta_n, beta_phi, gamma_n, gamma_phi (each has min/max)
  age,
  metric = c("pd", "ed", "nnd"),
  offset,
  n_per_param    = 10L,     # 10 values per parameter (inclusive)
  replicates     = 100L,    # trees per parameter combo
  max_total_sims = 100000L, # cap; set Inf to disable
  sample_seed    = 1L,      # controls grid sub-sampling (if capped) + RNG
  ncores         = max(1L, (parallel::detectCores(logical = FALSE) %||% 2L) - 1L),
  backend        = c("auto","multicore","psock"),  # "auto" picks psock on Windows, multicore elsewhere
  fix_gamma_zero = TRUE      # when TRUE, gamma_n=gamma_phi=0 and NOT swept in grid
) {
  `%||%` <- function(a,b) if (!is.null(a)) a else b
  metric  <- match.arg(metric)
  backend <- match.arg(backend)

  if (length(dists) != 5L)
    stop("dists must have 5 entries in this order: lambda, beta_n, beta_phi, gamma_n, gamma_phi")

  # 1) Parameter grids (10 points, including endpoints)
  lambda_grid    <- .mk_seq(dists[[1]], n_per_param)
  beta_n_grid    <- .mk_seq(dists[[2]], n_per_param)
  beta_phi_grid  <- .mk_seq(dists[[3]], n_per_param)

  if (fix_gamma_zero) {
    gamma_n_grid   <- 0
    gamma_phi_grid <- 0
  } else {
    gamma_n_grid   <- .mk_seq(dists[[4]], n_per_param)
    gamma_phi_grid <- .mk_seq(dists[[5]], n_per_param)
  }

  # 2) Full cartesian grid (shrinks automatically when gammas are fixed)
  grid <- expand.grid(
    lambda    = lambda_grid,
    beta_n    = beta_n_grid,
    beta_phi  = beta_phi_grid,
    gamma_n   = gamma_n_grid,
    gamma_phi = gamma_phi_grid,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  n_combos   <- nrow(grid)
  total_sims <- n_combos * replicates

  # 3) Optional cap on total simulations
  if (is.finite(max_total_sims) && total_sims > max_total_sims) {
    set.seed(sample_seed)
    keep_rows <- max(1L, ceiling(max_total_sims / replicates))
    message(sprintf("[%s] Grid has %d combos × %d replicates = %d sims.",
                    metric, n_combos, replicates, total_sims))
    message(sprintf("Capping to ~%d sims by sampling %d grid rows.", max_total_sims, keep_rows))
    sel <- sample.int(n_combos, size = keep_rows, replace = FALSE)
    grid <- grid[sel, , drop = FALSE]
    n_combos   <- nrow(grid)
    total_sims <- n_combos * replicates
  } else {
    message(sprintf("[%s] Grid has %d combos × %d replicates = %d sims.",
                    metric, n_combos, replicates, total_sims))
  }

  # 4) Choose backend
  if (backend == "auto") {
    backend <- if (.Platform$OS.type == "windows") "psock" else "multicore"
  }

  # Worker: simulate `replicates` trees for combo i
  simulate_combo <- function(i, grid, metric, age, offset, replicates) {
    lam    <- grid$lambda[i]
    beta_n <- grid$beta_n[i]
    beta_p <- grid$beta_phi[i]
    gam_n  <- if (is.null(grid$gamma_n)) 0 else grid$gamma_n[i]
    gam_p  <- if (is.null(grid$gamma_phi)) 0 else grid$gamma_phi[i]

    # Hard-enforce zeros if requested (even if grid carried values)
    if (isTRUE(fix_gamma_zero)) { gam_n <- 0; gam_p <- 0 }

    rows <- vector("list", length = replicates)
    kk <- 0L
    for (r in seq_len(replicates)) {
      mu <- stats::runif(1, min = 0, max = 0.8 * lam)
      pars_list <- c(lam, mu, beta_n, beta_p, gam_n, gam_p)
      raw <- evesim::edd_sim(
        pars   = pars_list,
        age    = as.double(age),
        metric = metric,
        offset = offset,
        size_limit = 2000
      )
      if (!is.null(raw$tes)) {
        tes <- raw$tes
        tas <- raw$tas
        kk <- kk + 1L
        rows[[kk]] <- tibble::tibble(
          tree_id   = sprintf("%s_%06d_rep%03d", metric, i, r),
          metric    = metric,
          tes       = list(tes),
          tas       = list(tas),
          lambda    = lam,
          mu        = mu,
          beta_n    = beta_n,
          beta_phi  = beta_p,
          gamma_n   = 0,      # record zeros in output
          gamma_phi = 0
        )
      }
    }
    if (kk == 0L) return(NULL)
    dplyr::bind_rows(rows[seq_len(kk)])
  }

  # 5) Parallel map over grid rows
  if (backend == "multicore") {
    set.seed(sample_seed)
    parts <- parallel::mclapply(
      X = seq_len(n_combos),
      FUN = simulate_combo,
      grid = grid, metric = metric, age = age, offset = offset, replicates = replicates,
      mc.cores = ncores, mc.preschedule = TRUE
    )
  } else {
    ncores <- max(1L, ncores)
    cl <- parallel::makeCluster(ncores)
    on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)

    parallel::clusterEvalQ(cl, {
      suppressPackageStartupMessages({
        library(evesim); library(ape); library(tibble); library(dplyr)
      })
      NULL
    })
    parallel::clusterSetRNGStream(cl, sample_seed)  # reproducible L'Ecuyer streams

    parallel::clusterExport(
      cl,
      varlist = c("grid","metric","age","offset","replicates","simulate_combo","fix_gamma_zero"),
      envir   = environment()
    )

    parts <- parallel::parLapply(
      cl = cl,
      X  = seq_len(n_combos),
      fun = simulate_combo,
      grid = grid, metric = metric, age = age, offset = offset, replicates = replicates
    )
  }

  # 6) Bind results
  parts <- parts[!vapply(parts, is.null, logical(1))]
  if (!length(parts)) {
    warning(sprintf("[%s] No trees were generated.", metric))
    return(tibble::tibble(
      tree_id = character(), metric = character(), tree = list(),
      lambda = numeric(), mu = numeric(), beta_n = numeric(), beta_phi = numeric(),
      gamma_n = numeric(), gamma_phi = numeric()
    ))
  }
  data.table::rbindlist(parts, fill = TRUE)
}

simulate_all_metrics_parallel <- function(
  dists_pd, dists_ed, dists_nnd,
  age, offset,
  n_per_param    = 5L,
  replicates     = 50L,
  max_total_sims = 100000L,
  sample_seed    = 1L,
  ncores         = max(1L, (parallel::detectCores(logical = FALSE) %||% 2L) - 1L),
  backend        = c("auto","multicore","psock"),
  fix_gamma_zero = TRUE
) {
  backend <- match.arg(backend)

  pd  <- simulate_metric_grid_parallel(dists_pd,  age, "pd",  offset,
                                       n_per_param, replicates,
                                       max_total_sims, sample_seed + 0L,
                                       ncores, backend, fix_gamma_zero)
  ed  <- simulate_metric_grid_parallel(dists_ed,  age, "ed",  "none",
                                       n_per_param, replicates,
                                       max_total_sims, sample_seed + 1L,
                                       ncores, backend, fix_gamma_zero)
  nnd <- simulate_metric_grid_parallel(dists_nnd, age, "nnd", "none",
                                       n_per_param, replicates,
                                       max_total_sims, sample_seed + 2L,
                                       ncores, backend, fix_gamma_zero)
  dplyr::bind_rows(pd, ed, nnd)
}


# ---------------------- Example usage (commented) -------------------------------
params <- yaml::read_yaml("config/eve_sim.yaml")

dists_pd <- params$dists_pd
dists_ed <- params$dists_ed
dists_nnd <- params$dists_nnd

# Manually set betas and gammas to zero, for 2-Pars or 4-pars simulation
# These lines set betas to zero
# dists_pd[[2]]$max <- 0
# dists_pd[[2]]$min <- 0
# dists_ed[[2]]$max <- 0
# dists_ed[[2]]$min <- 0
# dists_nnd[[2]]$max <- 0
# dists_nnd[[2]]$min <- 0
# dists_pd[[3]]$max <- 0
# dists_pd[[3]]$min <- 0
# dists_ed[[3]]$max <- 0
# dists_ed[[3]]$min <- 0
# dists_nnd[[3]]$max <- 0
# dists_nnd[[3]]$min <- 0

# These lines set gammas to zero
dists_pd[[4]]$max <- 0
dists_pd[[4]]$min <- 0
dists_ed[[4]]$max <- 0
dists_ed[[4]]$min <- 0
dists_nnd[[4]]$max <- 0
dists_nnd[[4]]$min <- 0
dists_pd[[5]]$max <- 0
dists_pd[[5]]$min <- 0
dists_ed[[5]]$max <- 0
dists_ed[[5]]$min <- 0
dists_nnd[[5]]$max <- 0
dists_nnd[[5]]$min <- 0

# Suppose df_all is your combined set across metrics:
# Assuming you’ve parsed those YAML-like blocks into R lists `dists_pd`, `dists_ed`, `dists_nnd`
# Each is a list of 5 elements with fields: distribution, n, min, max.

# Simulate in parallel
df_all <- simulate_all_metrics_parallel(
  dists_pd   = dists_pd,
  dists_ed   = dists_ed,
  dists_nnd  = dists_nnd,
  age        = 10,
  offset     = "simtime",
  n_per_param= 5,
  replicates = 50,
  max_total_sims = Inf,      # cap total work (raise or set Inf as needed)
  ncores = parallel::detectCores(logical = FALSE) - 1L,
  backend = "auto"              # "auto" picks psock on Windows, multicore elsewhere
)

# Store backup trees df
saveRDS(df_all, file.path(OUT_DIR, "simulated_trees.rds"))
# df_all has: tree_id, metric, tree (phylo), lambda, mu, beta_n, beta_phi, gamma_n, gamma_phi

ans <- analyze_forest(
  df_all,
  backend = "auto",
  unroll_mode = "grid",         # grid unrolling (uniform time slices)
  grid_k = 20L,                 # 20 time points, including ends (ends are skipped for slicing internally)
  grid_age_target = 10.0,       # align all trees to age 10
  grid_rescale_each_tree = TRUE # rescale each tree to 10
)
# Store backup analysis results
saveRDS(ans, file.path(OUT_DIR, "analysis_results.rds"))

# Save some plots:
DPI_HIGH <- 600
ggsave(file.path(OUT_DIR, "profiles_gaussian_nMGL.png"), plot_profiles_gauss(ans$full_gauss, "nMGL"), width=10, height=6, dpi=DPI_HIGH)
ggsave(file.path(OUT_DIR, "profiles_rff_by_h_nMGL.png"), plot_profiles_rff_by_h(ans$full_rff, "nMGL"), width=10, height=6, dpi=DPI_HIGH)
ggsave(file.path(OUT_DIR, "unrolled_heatmap_gaussian.png"), plot_unrolled_heatmap_gauss(ans$series_gauss), width=10, height=6, dpi=DPI_HIGH)
ggsave(file.path(OUT_DIR, "unrolled_heatmap_rff.png"), plot_unrolled_heatmap_rff(ans$series_rff), width=12, height=7, dpi=DPI_HIGH)
ggsave(file.path(OUT_DIR, "unrolled_ridgelines_rff.png"), plot_unrolled_ridgelines_rff(ans$series_rff), width=10, height=9, dpi=DPI_HIGH)

# Example JSD heatmap (RFF, h=0.1):
M <- ans$jsd_rff[["0.1"]]
if (!is.null(M)) ggsave(file.path(OUT_DIR, "JSD_RFF_h=0.1.png"), plot_jsd_heatmap(M, "JSD (RFF, nMGL, h=0.1)"), width=7, height=6, dpi=DPI_HIGH)
# =================================================================================
