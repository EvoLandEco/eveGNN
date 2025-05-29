args <- commandArgs(TRUE)

i    <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "DDD_MLE_TES/MLE_DATA/ddd_mle.rds"))

## -----------------------------------------------------------------
## Set working directories and create central DIAGNOSTICS folder
## -----------------------------------------------------------------
setwd(name)
setwd("DDD_MLE_TES")
root_dir   <- getwd()
diag_dir   <- file.path(root_dir, "DIAGNOSTICS")
if (!dir.exists(diag_dir)) dir.create(diag_dir)

## If data[[i]] has only two parameters, extend to three
if (length(data$pars[[i]]) == 2) data$pars[[i]][3] <- Inf

## -----------------------------------------------------------------
## Helper: run DDD::dd_ML once, capturing *all* console output
## -----------------------------------------------------------------
safe_ddml <- function(brts, initpars, opt_method, t_limit) {

  ## Capture both stdout and stderr/messages in one character vector
  log_con <- textConnection("runlog", open = "w", local = TRUE)
  sink(log_con, type = "output")
  sink(log_con, type = "message")

  ml <- NA
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
        methode     = "odeint::runge_kutta_cash_karp54"
      )
      if (length(ml) == 1 && is.na(ml)) ml <- NA
    }, timeout = t_limit),
    TimeoutException = function(ex) { ml <<- NA },
    error            = function(ex) { ml <<- NA }
  )

  sink(type = "message")
  sink(type = "output")
  close(log_con)

  list(ml = ml, log = runlog)
}

opt_methods <- c("simplex", "subplex", "DEoptim")

## -----------------------------------------------------------------
## Function to process one block (best or typical)
## -----------------------------------------------------------------
run_block <- function(block, brts, initpars, t_limit) {

  reps <- vector("list", length(opt_methods))

  for (k in seq_along(opt_methods)) {
    method <- opt_methods[k]
    res    <- safe_ddml(brts, initpars, method, t_limit)
    reps[[k]] <- res

    ## --------------------------------------------------------------
    ##  ➜ save diagnostics *immediately* if this replicate failed
    ## --------------------------------------------------------------
    if (is.na(res$ml[1])) {
      diag_file <- file.path(
        diag_dir,
        sprintf("diag_%s_%s_%d.rds", block, method, i)
      )
      diag_obj  <- list(
        block = block,
        opt_method = method,
        brts = brts,
        pars = initpars,
        log  = res$log
      )
      saveRDS(diag_obj, file = diag_file)
    }
  }

  ok_lls <- sapply(reps, function(x)
    if (is.numeric(x$ml)) x$ml[4] else NA_real_)

  if (all(is.na(ok_lls))) return(NA)
  best_j <- which.max(ok_lls)
  reps[[best_j]]$ml
}

## -----------------------------------------------------------------
## 1. Best-case block
## -----------------------------------------------------------------
best_ml <- run_block(
  block   = "best",
  brts    = data$brts[[i]],
  initpars = data$pars[[i]],
  t_limit = 30000
)

## -----------------------------------------------------------------
## 2. Typical-case block  (NO_INIT)
## -----------------------------------------------------------------
if (!dir.exists("NO_INIT")) dir.create("NO_INIT")
setwd("NO_INIT")

init_typical <- c(
  runif(1, 0.1, 4.0),
  runif(1, 0.0, 1.5),
  runif(1, 10.0, 1000.0)
)

typical_ml <- run_block(
  block   = "typical",
  brts    = data$brts[[i]],
  initpars = init_typical,
  t_limit = 30000
)

