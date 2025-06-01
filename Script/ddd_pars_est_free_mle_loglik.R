args <- commandArgs(TRUE)

i    <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "DDD_MLE_TES/MLE_DATA/ddd_mle.rds"))

setwd(name)
setwd("DDD_MLE_TES")

# If data i has only two parameters, set the third to Inf
if (length(data$pars[[i]]) == 2) data$pars[[i]][3] <- Inf

# --------------------------------------------------------------
# Helper: run one dd_ML call safely, returning the raw ml vector
# --------------------------------------------------------------
safe_ddml <- function(brts, initpars, opt_method, t_limit) {
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
        methode     = "odeint::runge_kutta_cash_karp54"
      )
      if (length(ml) == 1 && is.na(ml)) NA else ml
    }, timeout = t_limit),

    TimeoutException = function(ex) {
      info <- list(
        type       = "timeout",
        brts       = brts,
        initpars   = initpars,
        opt_method = opt_method,
        t_limit    = t_limit,
        error_msg  = ex$message
      )
      fname <- tempfile(
        pattern = paste0(opt_method, "_timeout_"),
        tmpdir  = err_dir,
        fileext = ".rds"
      )
      saveRDS(info, fname)
      NA
    },

    error = function(ex) {
      info <- list(
        type       = "error",
        brts       = brts,
        initpars   = initpars,
        opt_method = opt_method,
        t_limit    = t_limit,
        error_msg  = ex$message,
        call       = deparse(ex$call)
      )
      fname <- tempfile(
        pattern = paste0(opt_method, "_error_"),
        tmpdir  = err_dir,
        fileext = ".rds"
      )
      saveRDS(info, fname)
      NA
    }
  )
}

opt_methods <- c("simplex", "subplex", "DEoptim")

# --------------------------------------------------------------
# 1. Best-case block  (three different optimisation methods)
# --------------------------------------------------------------
best_reps <- vector("list", length(opt_methods))
for (k in seq_along(opt_methods)) {
  best_reps[[k]] <- safe_ddml(
    brts       = data$brts[[i]],
    initpars   = data$pars[[i]],
    opt_method = opt_methods[k],
    t_limit    = 30000
  )
}

best_lls <- sapply(best_reps, function(x) as.numeric(x[4]))

if (all(is.na(best_lls))) {
  best_out <- NA
} else {
  j         <- which.max(best_lls)
  ml        <- best_reps[[j]]
  names(ml) <- NULL
  best_out  <- eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])
  best_out$nnode  <- data$tes[[i]]$Nnode
}

saveRDS(best_out, file = paste0("differences_", i, ".rds"))


# --------------------------------------------------------------
# 2. Typical-case block (NO_INIT, three different methods)
# --------------------------------------------------------------
if (!dir.exists("NO_INIT")) dir.create("NO_INIT")
setwd("NO_INIT")

# One random initial vector, reused for all three methods
init_typical <- c(
  runif(1, 0.1, 4.0),
  runif(1, 0.0, 1.5),
  runif(1, 10.0, 1000.0)
)

typical_reps <- vector("list", length(opt_methods))
for (k in seq_along(opt_methods)) {
  typical_reps[[k]] <- safe_ddml(
    brts       = data$brts[[i]],
    initpars   = init_typical,
    opt_method = opt_methods[k],
    t_limit    = 30000
  )
}

typical_lls <- sapply(typical_reps, function(x) as.numeric(x[4]))

if (all(is.na(typical_lls))) {
  typical_out <- NA
} else {
  j          <- which.max(typical_lls)
  ml         <- typical_reps[[j]]
  names(ml)  <- NULL
  typical_out <- eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])
  typical_out$nnode  <- data$tes[[i]]$Nnode
}

saveRDS(typical_out, file = paste0("differences_", i, ".rds"))
