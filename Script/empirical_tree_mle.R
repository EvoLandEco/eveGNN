args <- commandArgs(TRUE)

name <- as.character(args[1])

setwd(name)

### Condamine 2019 Ecology Letters Trees Loading
load("EMP_DATA/FamilyAmphibiaTrees.Rdata")
load("EMP_DATA/FamilyBirdTrees.Rdata")
load("EMP_DATA/FamilyCrocoTurtleTrees.Rdata")
load("EMP_DATA/FamilyMammalTrees.Rdata")
load("EMP_DATA/FamilySquamateTrees.Rdata")

load("D:\\Data\\Empirical Trees\\Condamine2019\\FamilyAmphibiaTrees.Rdata")
load("D:\\Data\\Empirical Trees\\Condamine2019\\FamilyBirdTrees.Rdata")
load("D:\\Data\\Empirical Trees\\Condamine2019\\FamilyCrocoTurtleTrees.Rdata")
load("D:\\Data\\Empirical Trees\\Condamine2019\\FamilyMammalTrees.Rdata")
load("D:\\Data\\Empirical Trees\\Condamine2019\\FamilySquamateTrees.Rdata")


condamine_tree_list <- list(Amphibia = FamilyAmphibiaTrees,
                            Bird = FamilyBirdTrees,
                            CrocoTurtle = FamilyCrocoTurtleTrees,
                            Mammal = FamilyMammalTrees,
                            Squamate = FamilySquamateTrees)

if (!dir.exists("EMP_RESULT")) {
  dir.create("EMP_RESULT")
}

setwd("EMP_RESULT")

if (!dir.exists("DDD")) {
  dir.create("DDD")
}

setwd("DDD")

family_list <- names(condamine_tree_list)

for (i in 1:length(family_list)) {
  family_name <- family_list[i]
  tree_list <- names(condamine_tree_list[[family_name]])
  for (j in 1:length(tree_list)) {
    tree_name <- tree_list[j]
    meta <- c("Family" = family_name, "Tree" = tree_name)
    tree <- condamine_tree_list[[family_name]][[tree_name]]$tree
    tree <- eveGNN::rescale_crown_age(tree, 10)
    tree_brts <- treestats::branching_times(tree)
    tree_brts <- sort(tree_brts, decreasing = TRUE)
    file_name <- paste0("tree_brts_", family_name, "_", tree_name, ".rds")
    saveRDS(tree_brts, file_name)
    system(paste0("sbatch ../../../../Bash/submit_ddd_emp_mle.sh ", paste0(file_name, " ", family_name, " ", tree_name)))
  }
}

setwd("..")

if (!dir.exists("BD")) {
  dir.create("BD")
}

setwd("BD")

for (i in 1:length(family_list)) {
  family_name <- family_list[i]
  tree_list <- names(condamine_tree_list[[family_name]])
  for (j in 1:length(tree_list)) {
    tree_name <- tree_list[j]
    meta <- c("Family" = family_name, "Tree" = tree_name)
    tree <- condamine_tree_list[[family_name]][[tree_name]]$tree
    tree <- eveGNN::rescale_crown_age(tree, 10)
    tree_brts <- treestats::branching_times(tree)
    tree_brts <- sort(tree_brts, decreasing = TRUE)
    ml <- DDD::bd_ML(
      brts = tree_brts,
      idparsopt = c(1, 2),
      tdmodel = 0,
      btorph = 0,
      soc = 2,
      cond = 1,
      num_cycles = Inf
    )
    df_bd_results <- data.frame(Family = family_name,
                                Tree = tree_name,
                                lambda = ml$lambda0,
                                mu = ml$mu0,
                                loglik=ml$loglik,
                                df=ml$df,
                                conv=ml$conv)
    saveRDS(df_bd_results, file = paste0("BD_EMP_MLE_", family_name, "_", tree_name, ".rds"))
  }
}

setwd("..")

if (!dir.exists("PBD")) {
  dir.create("PBD")
}

setwd("PBD")

for (i in 1:length(family_list)) {
  family_name <- family_list[i]
  tree_list <- names(condamine_tree_list[[family_name]])
  for (j in 1:length(tree_list)) {
    tree_name <- tree_list[j]
    meta <- c("Family" = family_name, "Tree" = tree_name)
    tree <- condamine_tree_list[[family_name]][[tree_name]]$tree
    tree <- eveGNN::rescale_crown_age(tree, 10)
    tree_brts <- treestats::branching_times(tree)
    tree_brts <- sort(tree_brts, decreasing = TRUE)
    ml <- PBD::pbd_ML(
      brts = tree_brts,
      initparsopt = c(0.2, 0.1, 1, 0.1),
      idparsopt = 1:4,
      exteq = 0,
      btorph = 0,
      soc = 2,
      verbose = FALSE
    )
    df_pbd_results <- data.frame(Family = family_name,
                                 Tree = tree_name,
                                 b1 = ml$b,
                                 lambda1 = ml$lambda_1,
                                 b2 = ml$b,
                                 mu1 = ml$mu_1,
                                 mu2 = ml$mu_2,
                                 loglik=ml$loglik,
                                 df=ml$df,
                                 conv=ml$conv)
    saveRDS(df_pbd_results, file = paste0("PBD_EMP_MLE_", family_name, "_", tree_name, ".rds"))
  }
}
