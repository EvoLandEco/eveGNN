args <- commandArgs(TRUE)

family_name <- as.character(args[1])
tree_name <- as.character(args[2])
lambda <- as.numeric(args[3])
mu <- as.numeric(args[4])
cap <- as.numeric(args[5])
index <- as.numeric(args[6])

dir.create("MLE_UNCERTAINTY", showWarnings = FALSE)

setwd("MLE_UNCERTAINTY")

sim <- DDD::dd_sim(c(lambda, mu, cap), age = 10, ddmodel = 1)

brts <- sim$brts

ml <- DDD::dd_ML(
  brts = brts,
  idparsopt = c(1, 2, 3),
  btorph = 0,
  soc = 2,
  cond = 1,
  ddmodel = 1,
  num_cycles = Inf,
  optimmethod = 'simplex'
)

df_ddd_results <- data.frame(Family = family_name,
                             Tree = tree_name,
                             lambda = ml$lambda,
                             mu = ml$mu,
                             cap = ml$K,
                             loglik=ml$loglik,
                             df=ml$df,
                             conv=ml$conv)

saveRDS(df_ddd_results, file = paste0("DDD_EMP_MLE_UNCERTAINTY_", family_name, "_", tree_name, "_", index, ".rds"))
