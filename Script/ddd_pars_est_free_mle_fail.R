args <- commandArgs(TRUE)
i    <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "DDD_MLE_TES/MLE_DATA/ddd_mle.rds"))

setwd(file.path(name, "DDD_MLE_TES"))
if (length(data$pars[[i]]) == 2) data$pars[[i]][3] <- Inf

diag_dir <- "DIAGNOSTICS"
if (!dir.exists(diag_dir)) dir.create(diag_dir)

safe_ddml <- function(brts, initpars, opt_method, t_limit) {
  log_con <- textConnection("runlog", "w", local = TRUE)
  sink(log_con, type = "output")
  sink(log_con, type = "message")
  ml <- tryCatch(
    R.utils::withTimeout({
      res <- DDD::dd_ML(
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
      if (length(res) == 1 && is.na(res)) NA else res
    }, timeout = t_limit),
    TimeoutException = function(e) NA,
    error            = function(e) NA
  )
  sink(type = "message"); sink(type = "output"); close(log_con)
  list(ml = ml, log = runlog)
}

run_block <- function(block, brts, initpars, truepars, t_limit) {
  methods <- c("simplex", "subplex", "DEoptim")
  reps <- lapply(methods, \(m) safe_ddml(brts, initpars, m, t_limit))

  for (k in seq_along(reps)) {
    if (is.na(reps[[k]]$ml[1])) {
      saveRDS(
        list(
          block      = block,
          opt_method = methods[k],
          brts       = brts,
          pars       = truepars,
          log        = reps[[k]]$log
        ),
        file = file.path(
          diag_dir,
          sprintf("diag_%s_%s_%d.rds", block, methods[k], i)
        )
      )
    }
  }

  lls <- sapply(reps, \(x) if (is.numeric(x$ml)) x$ml[4] else NA_real_)
  if (all(is.na(lls))) NA else reps[[which.max(lls)]]$ml
}

best_ml <- run_block(
  "best",
  data$brts[[i]],
  data$pars[[i]],
  data$pars[[i]],
  30000
)

if (!dir.exists("NO_INIT")) dir.create("NO_INIT")
setwd("NO_INIT")

init_typical <- c(runif(1, 0.1, 4.0), runif(1, 0.0, 1.5), runif(1, 10, 1000))
typical_ml <- run_block(
  "typical",
  data$brts[[i]],
  init_typical,
  data$pars[[i]],
  30000
)
