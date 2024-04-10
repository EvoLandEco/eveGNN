args <- commandArgs(TRUE)

name <- as.character(args[1])

data_path <- file.path(name, "EMP_DATA")
export_path <- file.path(name, "EMP_DATA", "EXPORT")

if (!dir.exists(data_path)) {
  stop("Empirical data path does not exist")
}

if (!dir.exists(export_path)) {
  dir.create(export_path, recursive = TRUE)
}

### Read OTT trees
ott_trees <- readRDS(file.path(data_path, "OTT_trees.rds"))

for (i in 1:length(ott_trees)) {
  tree <- ott_trees[[i]]
  tree <- eveGNN::rescale_crown_age(tree, 10)
  eveGNN::export_to_gnn_ott(tree, i, export_path)
}
