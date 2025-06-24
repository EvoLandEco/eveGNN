args <- commandArgs(TRUE)
name    <- as.numeric(args[1])

data <- readRDS(file.path(name, "DDD_MLE_TES/differences_1187.rds"))

DDD::dd_ML(brts=data$input$brts,
           initparsopt = data$input$initpars,
           idparsopt   = c(1, 2, 3),
           btorph      = 0,
           soc         = 2,
           cond        = 1,
           ddmodel     = 1,
           num_cycles  = Inf,
           optimmethod = "simplex",
           methode="odeint::runge_kutta_cash_karp54")
