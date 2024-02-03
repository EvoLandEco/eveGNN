args <- commandArgs(TRUE)

name <- as.character(args[1])

boot_path <- file.path(name, "BOOTSTRAP")

if (!dir.exists(boot_path)) {
  dir.create(boot_path, recursive = TRUE)
}

## load DDD gnn emp results
ddd_gnn_emp <- readRDS(file.path(name, "EMP_RESULT", "DDD", "DDD_EMP_GNN_predictions.rds"))
ddd_gnn_emp <- as.data.frame(ddd_gnn_emp)
ddd_path <- file.path(boot_path, "DDD")

if (!dir.exists(ddd_path)) {
  dir.create(ddd_path, recursive = TRUE)
}

for (i in 1:nrow(ddd_gnn_emp)) {
  lambda <- ddd_gnn_emp$lambda_pred[i]
  mu <- ddd_gnn_emp$mu_pred[i]
  cap <- ddd_gnn_emp$cap_pred[i]
  # number of tips, computed from total number of nodes
  ntip <- (ddd_gnn_emp$nodes[i] - 1) / 2 + 1
  family_name <- ddd_gnn_emp$family[i]
  tree_name <- ddd_gnn_emp$tree[i]

  command <- paste0("sbatch submit_ddd_gnn_bootstrap.sh ",
                    paste0(lambda, " ", mu, " ", cap, " ", ntip, " ", family_name, " ", tree_name, " ", ddd_path))
}
