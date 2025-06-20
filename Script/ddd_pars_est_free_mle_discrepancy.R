# ------------------------------------------------------------------
# MLE pipeline – stores, per data set, *all* optimiser / integrator
# results *plus* pair-wise discrepancies between them.  One RDS file
# per data-set (= i) is written for “best-start” and “typical-start”.
# ------------------------------------------------------------------

args <- commandArgs(TRUE)
i    <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name,
                          "DDD_MLE_TES",
                          "MLE_DATA",
                          "ddd_mle.rds"))

setwd(file.path(name, "DDD_MLE_TES"))
if (length(data$pars[[i]]) == 2) data$pars[[i]][3] <- Inf

# ------------------------------------------------------------------
# Safe wrapper around DDD::dd_ML  – logs timeouts / errors
# ------------------------------------------------------------------
safe_ddml <- function(brts, initpars, opt_method, int_method, t_limit) {

  err_dir <- "safe_ddml_errors"
  if (!dir.exists(err_dir)) dir.create(err_dir)

  tryCatch(
    R.utils::withTimeout({
      ml <- DDD::dd_ML(
        brts        = brts,
        initparsopt = initpars,
        idparsopt   = c(1, 2, 3),
        btorph      = 0,
        soc         = 2,
        cond        = 1,
        ddmodel     = 1,
        num_cycles  = Inf,
        optimmethod = opt_method,
        methode     = int_method
      )
      if (length(ml) == 1 && is.na(ml)) NA else ml
    }, timeout = t_limit),

    TimeoutException = function(ex) {
      saveRDS(list(type = "timeout",
                   brts = brts,
                   initpars = initpars,
                   opt_method = opt_method,
                   int_method = int_method,
                   msg = ex$message),
              tempfile(pattern = sprintf("%s_%s_timeout_",
                                         opt_method,
                                         basename(int_method)),
                       tmpdir  = err_dir,
                       fileext = ".rds"))
      NA
    },

    error = function(ex) {
      saveRDS(list(type = "error",
                   brts = brts,
                   initpars = initpars,
                   opt_method = opt_method,
                   int_method = int_method,
                   msg = ex$message),
              tempfile(pattern = sprintf("%s_%s_error_",
                                         opt_method,
                                         basename(int_method)),
                       tmpdir  = err_dir,
                       fileext = ".rds"))
      NA
    }
  )
}

# ------------------------------------------------------------------
# Optimiser / integrator combinations
# ------------------------------------------------------------------
opt_methods <- c("simplex")
int_methods <- c("analytical", "odeint::runge_kutta_cash_karp54")
combos      <- expand.grid(opt = opt_methods,
                           int = int_methods,
                           KEEP.OUT.ATTRS = FALSE,
                           stringsAsFactors = FALSE)

# ------------------------------------------------------------------
# Run one block (best-start or typical-start) and save everything
# ------------------------------------------------------------------
run_block <- function(block_dir, brts, initpars, t_limit) {

  if (!dir.exists(block_dir)) dir.create(block_dir, recursive = TRUE)

  results <- vector("list", nrow(combos))

  for (k in seq_len(nrow(combos))) {
    om <- combos$opt[k]
    im <- combos$int[k]
    ml <- safe_ddml(brts, initpars, om, im, t_limit)

    if (length(ml) == 1 && is.na(ml)) {
      results[[k]] <- list(opt_method  = om,
                           integrator  = im,
                           loglik      = NA_real_,
                           est         = rep(NA_real_, 3),
                           abs_err     = rep(NA_real_, 3),
                           rel_err_pc  = rep(NA_real_, 3))
    } else {
      est   <- as.numeric(ml[1:3])
      true  <- data$pars[[i]]
      abs_e <- true - est
      rel_e <- 100 * abs_e / true

      results[[k]] <- list(opt_method  = om,
                           integrator  = im,
                           loglik      = ml[4],
                           est         = est,
                           abs_err     = abs_e,
                           rel_err_pc  = rel_e)
    }
  }

  # ----------------------------------------------------------------
  # Pair-wise discrepancies between method combinations
  # ----------------------------------------------------------------
  comb_pairs <- combn(seq_along(results), 2, simplify = FALSE)

  discrep <- lapply(comb_pairs, function(idx) {
    a <- results[[idx[1]]]
    b <- results[[idx[2]]]

    if (anyNA(c(a$est, b$est)))
      return(NULL)

    list(
      methodA   = paste(a$opt_method, a$integrator, sep = "|"),
      methodB   = paste(b$opt_method, b$integrator, sep = "|"),
      abs_diff  = a$est - b$est,
      rel_diff  = 100 * (a$est - b$est) / a$est        # % of A
    )
  })
  discrep <- Filter(Negate(is.null), discrep)

  # ----------------------------------------------------------------
  # Build output object and write RDS
  # ----------------------------------------------------------------
  out <- list(
    input  = list(brts = brts,
                  initpars = initpars,
                  truepars = data$pars[[i]]),
    results = results,
    pairwise_discrep = discrep
  )

  saveRDS(out, file.path(block_dir,
                         sprintf("differences_%d.rds", i)))
}

# ------------------------------------------------------------------
# 1. TRUE-START block
# ------------------------------------------------------------------
run_block(block_dir = ".",
          brts      = data$brts[[i]],
          initpars  = data$pars[[i]],
          t_limit   = 43200)

# ------------------------------------------------------------------
# 2. RANDOM-START block
# ------------------------------------------------------------------
if (!dir.exists("NO_INIT")) dir.create("NO_INIT")
setwd("NO_INIT")

init_typ <- c(runif(1, 0.1, 4.0),
              runif(1, 0.0, 1.5),
              runif(1, 10.0, 1000.0))

run_block(block_dir = ".",
          brts      = data$brts[[i]],
          initpars  = init_typ,
          t_limit   = 43200)
