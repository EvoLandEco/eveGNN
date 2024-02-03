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

### Condamine 2019 Ecology Letters Trees Loading
load("EMP_DATA/FamilyAmphibiaTrees.Rdata")
load("EMP_DATA/FamilyBirdTrees.Rdata")
load("EMP_DATA/FamilyCrocoTurtleTrees.Rdata")
load("EMP_DATA/FamilyMammalTrees.Rdata")
load("EMP_DATA/FamilySquamateTrees.Rdata")

condamine_tree_list <- list(Amphibia = FamilyAmphibiaTrees,
                            Bird = FamilyBirdTrees,
                            CrocoTurtle = FamilyCrocoTurtleTrees,
                            Mammal = FamilyMammalTrees,
                            Squamate = FamilySquamateTrees)

family_list <- names(condamine_tree_list)

for (i in 1:length(family_list)) {
  family_name <- family_list[i]
  tree_list <- names(condamine_tree_list[[family_name]])
  for (j in 1:length(tree_list)) {
    tree_name <- tree_list[j]
    meta <- c("Family" = family_name, "Tree" = tree_name)
    tree <- condamine_tree_list[[family_name]][[tree_name]]$tree
    tree <- eveGNN::rescale_crown_age(tree, 10)
    eveGNN::export_to_gnn_empirical(tree, meta, export_path)
  }
}