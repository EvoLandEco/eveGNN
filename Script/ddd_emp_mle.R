args <- commandArgs(TRUE)

file_name <- as.character(args[1])

family_name <- as.character(args[2])

tree_name <- as.character(args[3])

tree_brts <- readRDS(file_name)

ml <- DDD::dd_ML(
  brts = tree_brts,
  idparsopt = c(1, 2, 3),
  btorph = 0,
  soc = 2,
  cond = 1,
  ddmodel = 1,
  num_cycles = 1
)

df_ddd_results <- data.frame(Family = family_name,
                             Tree = tree_name,
                             lambda = ml$lambda,
                             mu = ml$mu,
                             cap = ml$K,
                             loglik=ml$loglik,
                             df=ml$df,
                             conv=ml$conv)

saveRDS(df_ddd_results, file = paste0("DDD_EMP_MLE_", family_name, "_", tree_name, ".rds"))
