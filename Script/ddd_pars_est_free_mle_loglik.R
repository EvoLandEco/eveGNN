args <- commandArgs(TRUE)
i    <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "DDD_MLE_TES/MLE_DATA/ddd_mle.rds"))

setwd(file.path(name, "DDD_MLE_TES"))
if (length(data$pars[[i]]) == 2) data$pars[[i]][3] <- Inf

## ----------------------------------------------------------------
## Safe wrapper
## ----------------------------------------------------------------
safe_ddml <- function(brts, initpars, opt_method, int_method, t_limit) {
  err_dir <- "safe_ddml_errors"
  if (!dir.exists(err_dir)) dir.create(err_dir, recursive = TRUE)

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
      info <- list(
        type        = "timeout",
        brts        = brts,
        initpars    = initpars,
        opt_method  = opt_method,
        int_method  = int_method,
        t_limit     = t_limit,
        error_msg   = ex$message
      )
      saveRDS(info,
              tempfile(
                pattern = sprintf("%s_%s_timeout_", opt_method, basename(int_method)),
                tmpdir  = err_dir,
                fileext = ".rds"))
      NA
    },

    error = function(ex) {
      info <- list(
        type        = "error",
        brts        = brts,
        initpars    = initpars,
        opt_method  = opt_method,
        int_method  = int_method,
        t_limit     = t_limit,
        error_msg   = ex$message,
        call        = deparse(ex$call)
      )
      saveRDS(info,
              tempfile(
                pattern = sprintf("%s_%s_error_", opt_method, basename(int_method)),
                tmpdir  = err_dir,
                fileext = ".rds"))
      NA
    }
  )
}

opt_methods  <- c("simplex")
int_methods  <- c("odeint::runge_kutta_cash_karp54")
combos       <- expand.grid(opt = opt_methods, int = int_methods, stringsAsFactors = FALSE)

## ----------------------------------------------------------------
## Helper to run *one* block (best or typical) and save everything
## ----------------------------------------------------------------
run_block <- function(block_dir, brts, initpars, t_limit) {
  if (!dir.exists(block_dir)) dir.create(block_dir, recursive = TRUE)

  results <- vector("list", nrow(combos))

  for (row in seq_len(nrow(combos))) {
    om <- combos$opt[row]
    im <- combos$int[row]
    ml <- safe_ddml(brts, initpars, om, im, t_limit)

    if (length(ml) == 1 && is.na(ml)) {
      results[[row]] <- list(
        opt_method = om,
        methode    = im,
        loglik     = NA_real_,
        mle        = NA,
        differences = NA,
        nnode      = data$tes[[i]]$Nnode
      )
    } else {
      diffs <- eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])
      results[[row]] <- list(
        opt_method  = om,
        methode     = im,
        loglik      = as.numeric(ml[4]),
        mle         = ml[1:3],
        differences = diffs,
        nnode       = data$tes[[i]]$Nnode
      )
    }
  }

  saveRDS(results, file.path(block_dir, sprintf("differences_%d.rds", i)))
}

## ----------------------------------------------------------------
## 1.  BEST-CASE  (true parameters as starting point)
## ----------------------------------------------------------------
run_block(
  block_dir = ".",
  brts      = data$brts[[i]],
  initpars  = data$pars[[i]],
  t_limit   = 43200
)

## ----------------------------------------------------------------
## 2.  TYPICAL-CASE  (random starting point)
## ----------------------------------------------------------------
if (!dir.exists("NO_INIT")) dir.create("NO_INIT")
setwd("NO_INIT")

init_typical <- c(runif(1, 0.1, 4.0),
                  runif(1, 0.0, 1.5),
                  runif(1, 10.0, 1000.0))

run_block(
  block_dir = ".",
  brts      = data$brts[[i]],
  initpars  = init_typical,
  t_limit   = 43200
)
