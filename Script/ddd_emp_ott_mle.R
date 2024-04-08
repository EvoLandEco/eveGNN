args <- commandArgs(TRUE)

i <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "EMP_DATA/OTT_trees.rds"))

setwd(name)
setwd("DDD_EMP_MLE")

tree <- data[[i]]
tree <- eveGNN::rescale_crown_age(tree, 10)
brts <- sort(treestats::branching_times(tree), decreasing = TRUE)

ml <- DDD::dd_ML(
  brts = brts,
  idparsopt = c(1, 2, 3),
  btorph = 0,
  soc = 2,
  cond = 1,
  ddmodel = 1,
  optimmethod = 'simplex'
)

ott_results <- data.frame(index = i,
                          lambda = ml$lambda,
                          mu = ml$mu,
                          cap = ml$K,
                          loglik = ml$loglik,
                          df = ml$df,
                          conv = ml$conv)

saveRDS(ott_results, file = paste0("DDD_EMP_MLE_OTT_", i, ".rds"))