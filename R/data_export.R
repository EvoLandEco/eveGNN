#' @export export_to_gnn
export_to_gnn <- function(data, name, which = "tas", undirected = FALSE) {
  path <- file.path(paste0("set_", name), "GNN/tree/")
  path_EL <- file.path(paste0("set_", name), "GNN/tree/EL/")
  eve:::check_path(path)
  eve:::check_path(path_EL)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      file_name <- paste0(path, "/tree_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      file_name <- paste0(path_EL, "/EL_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]]), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      file_name <- paste0(path, "/tree_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      file_name <- paste0(path_EL, "/EL_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]]), file = file_name)
    }
  }
}


#' @export export_to_gnn_empirical
export_to_gnn_empirical <- function(data, meta, path, undirected = FALSE) {
  path <- file.path(path, "GNN/tree/")
  path_EL <- file.path(path, "EL/")
  eve:::check_path(path)
  eve:::check_path(path_EL)

  file_name <- paste0(path, "/tree_", meta["Family"], "_", meta["Tree"], ".rds")
  saveRDS(tree_to_connectivity(data, undirected = undirected), file = file_name)

  file_name_el <- paste0(path_EL, "/EL_", meta["Family"], "_", meta["Tree"], ".rds")
  saveRDS(tree_to_adj_mat(data), file = file_name_el)
}


#' @export export_to_gnn_ott
export_to_gnn_ott <- function(data, index, path, undirected = FALSE) {
  path <- file.path(path, "GNN/tree/")
  path_EL <- file.path(path, "EL/")
  path_ST <- file.path(path, "ST/")
  path_BT <- file.path(path, "BT/")
  eve:::check_path(path)
  eve:::check_path(path_EL)
  eve:::check_path(path_ST)
  eve:::check_path(path_BT)

  file_name <- paste0(path, "/tree_", index, ".rds")
  saveRDS(tree_to_connectivity(data, undirected = undirected), file = file_name)

  file_name_el <- paste0(path_EL, "/EL_", index, ".rds")
  saveRDS(tree_to_adj_mat(data), file = file_name_el)

  file_name_st <- paste0(path_ST, "/ST_", index, ".rds")
  saveRDS(tree_to_stats(data), file = file_name_st)

  file_name_bt <- paste0(path_BT, "/BT_", index, ".rds")
  saveRDS(tree_to_brts(data), file = file_name_bt)
}


#' @export export_to_gnn_bootstrap
export_to_gnn_bootstrap <- function(data, meta, index, path, undirected = FALSE) {
  path <- file.path(path, "GNN/tree/")
  path_EL <- file.path(path, "EL/")
  eve:::check_path(path)
  eve:::check_path(path_EL)

  file_name <- paste0(path, "/tree_", meta["Family"], "_", meta["Tree"], "_", index, ".rds")
  saveRDS(tree_to_connectivity(data, undirected = undirected), file = file_name)

  file_name_el <- paste0(path_EL, "/EL_", meta["Family"], "_", meta["Tree"], "_", index, ".rds")
  saveRDS(tree_to_adj_mat(data), file = file_name_el)
}

# This function is for DDD
# TODO: Combine several functions into one loop
#' @export export_to_gnn_with_params
export_to_gnn_with_params <- function(data, which = "tas", undirected = FALSE, master = FALSE) {
  path <- file.path("GNN/tree/")
  path_EL <- file.path("GNN/tree/EL/")
  path_ST <- file.path("GNN/tree/ST/")
  path_BT <- file.path("GNN/tree/BT/")
  eve:::check_path(path)
  eve:::check_path(path_EL)
  eve:::check_path(path_ST)
  eve:::check_path(path_BT)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected, master = master), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]], master = master), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_ST, "/ST_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_stats(data$tas[[i]]), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_BT, "/BT_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_brts(data$tas[[i]]), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected, master = master), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]], master = master), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_ST, "/ST_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_stats(data$tes[[i]]), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_BT, "/BT_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_brts(data$tes[[i]]), file = file_name)
    }
  }
}


#' @export export_to_gnn_with_params_bd
export_to_gnn_with_params_bd <- function(data, which = "tas", undirected = FALSE) {
  path <- file.path("GNN/tree/")
  path_EL <- file.path("GNN/tree/EL/")
  path_ST <- file.path("GNN/tree/ST/")
  path_BT <- file.path("GNN/tree/BT/")
  eve:::check_path(path)
  eve:::check_path(path_EL)
  eve:::check_path(path_ST)
  eve:::check_path(path_BT)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]]), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path_ST, "/ST_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_stats(data$tas[[i]]), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path_BT, "/BT_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_brts(data$tas[[i]]), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]]), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path_ST, "/ST_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_stats(data$tes[[i]]), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      age <- data$age[[i]]
      file_name <- paste0(path_BT, "/BT_", la, "_", mu, "_", age, "_", i, ".rds")
      saveRDS(tree_to_brts(data$tes[[i]]), file = file_name)
    }
  }
}


#' @export export_to_gnn_with_params_eve
export_to_gnn_with_params_eve <- function(data, which = "tas", undirected = FALSE) {
  path <- file.path("GNN/tree/")
  path_EL <- file.path("GNN/tree/EL/")
  eve:::check_path(path)
  eve:::check_path(path_EL)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      beta_n <- data$pars[[i]][3]
      beta_phi <- data$pars[[i]][4]
      age <- data$age[[i]]
      metric <- data$metric[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", beta_n, "_", beta_phi, "_", age, "_", metric, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      beta_n <- data$pars[[i]][3]
      beta_phi <- data$pars[[i]][4]
      age <- data$age[[i]]
      metric <- data$metric[[i]]
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", beta_n, "_", beta_phi, "_", age, "_", metric, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]]), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      beta_n <- data$pars[[i]][3]
      beta_phi <- data$pars[[i]][4]
      age <- data$age[[i]]
      metric <- data$metric[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", beta_n, "_", beta_phi, "_", age, "_", metric, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      beta_n <- data$pars[[i]][3]
      beta_phi <- data$pars[[i]][4]
      age <- data$age[[i]]
      metric <- data$metric[[i]]
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", beta_n, "_", beta_phi, "_", age, "_", metric, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]]), file = file_name)
    }
  }
}


#' @export export_to_gnn_with_params_pbd
export_to_gnn_with_params_pbd <- function(data, which = "tes", undirected = FALSE) {
  path <- file.path("GNN/tree/")
  path_EL <- file.path("GNN/tree/EL/")
  path_ST <- file.path("GNN/tree/ST/")
  path_BT <- file.path("GNN/tree/BT/")
  eve:::check_path(path)
  eve:::check_path(path_EL)
  eve:::check_path(path_ST)
  eve:::check_path(path_BT)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path_EL, "/EL_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]]), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path_ST, "/ST_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_stats(data$tas[[i]]), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path_BT, "/BT_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_brts(data$tas[[i]]), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path_EL, "/EL_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]]), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path_ST, "/ST_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_stats(data$tes[[i]]), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la1 <- data$pars[[i]][1]
      la2 <- data$pars[[i]][2]
      la3 <- data$pars[[i]][3]
      mu1 <- data$pars[[i]][4]
      mu2 <- data$pars[[i]][5]
      age <- data$age[[i]]
      file_name <- paste0(path_BT, "/BT_", la1, "_", la2, "_", la3, "_", mu1, "_", mu2, "_", age, "_", i, ".rds")
      saveRDS(tree_to_brts(data$tes[[i]]), file = file_name)
    }
  }
}


#' @export export_to_gnn_batch
export_to_gnn_batch <- function(data, name, batch, batch_size, which = "tas", undirected = FALSE) {
  path <- file.path(paste0("set_", name), "GNN/tree/")
  path_EL <- file.path(paste0("set_", name), "GNN/tree/EL/")
  eve:::check_path(path)
  eve:::check_path(path_EL)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      file_name <- paste0(path, "/tree_", (batch - 1) * batch_size + i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      file_name <- paste0(path_EL, "/EL_", (batch - 1) * batch_size + i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]]), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      file_name <- paste0(path, "/tree_", (batch - 1) * batch_size + i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      file_name <- paste0(path_EL, "/EL_", (batch - 1) * batch_size + i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]]), file = file_name)
    }
  }
}


# TODO: Check unused parameter data, safely remove it if possible
#' @export write_pars_to_gnn
write_pars_to_gnn <- function(data, name) {
  index <- as.numeric(as.character(name))
  path <- file.path(paste0("set_", name), "GNN/")
  eve:::check_path(path)

  params <- read.table("params.txt")
  write.table(params[index, ], file.path(path, "params.txt"))
}


#' @export get_all_neighbors
get_all_neighbors <- function(tree) {
  # Initialize an empty list to store neighbors
  all_neighbors <- vector("list", Nnode(tree) + Ntip(tree))

  # Iterate through the edge matrix
  for (i in seq_len(nrow(tree$edge))) {
    from_node <- tree$edge[i, 1]
    to_node <- tree$edge[i, 2]

    # Append to_node to the neighbor list of from_node
    all_neighbors[[from_node]] <- c(all_neighbors[[from_node]], to_node)

    # Append from_node to the neighbor list of to_node
    all_neighbors[[to_node]] <- c(all_neighbors[[to_node]], from_node)
  }

  # Name the list elements for clarity
  names(all_neighbors) <- 1:(ape::Nnode(tree) + ape::Ntip(tree))

  return(all_neighbors)
}


#' @export get_all_neighbors_distances
get_all_neighbors_distances <- function(tree) {
  # Initialize an empty list to store neighbors and edge lengths
  all_neighbors <- vector("list", ape::Nnode(tree) + ape::Ntip(tree))

  # Iterate through the edge matrix
  for (i in seq_len(nrow(tree$edge))) {
    from_node <- tree$edge[i, 1]
    to_node <- tree$edge[i, 2]
    edge_length <- tree$edge.length[i]

    # Append to_node and edge_length to the neighbor list of from_node
    if (is.null(all_neighbors[[from_node]])) {
      all_neighbors[[from_node]] <- setNames(vector("numeric", 0), character(0))
    }
    all_neighbors[[from_node]][as.character(to_node)] <- edge_length

    # Append from_node and edge_length to the neighbor list of to_node
    if (is.null(all_neighbors[[to_node]])) {
      all_neighbors[[to_node]] <- setNames(vector("numeric", 0), character(0))
    }
    all_neighbors[[to_node]][as.character(from_node)] <- edge_length
  }

  # Name the list elements for clarity
  names(all_neighbors) <- 1:(ape::Nnode(tree) + ape::Ntip(tree))

  return(all_neighbors)
}


#' @export tree_to_adj_mat
tree_to_adj_mat <- function(tree, master = FALSE) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  neighbor_dists <- get_all_neighbors_distances(tree)

  padded_dists <- lapply(neighbor_dists, function(x) {
    if (length(x) == 1) {
      x <- c(x, 0, 0)  # Add two zeros after if length is 1
    }
    if (length(x) == 2) {
      x <- c(0, x)  # Add one zero before if length is 2
    }
    return(x)
  })

  neighbor_matrix <- do.call(rbind, padded_dists)
  colnames(neighbor_matrix) <- NULL

  if (master) {
    neighbor_matrix <- rbind(neighbor_matrix, rep(0, ncol(neighbor_matrix)))
  }

  return(neighbor_matrix)
}


#' @export tree_to_connectivity
tree_to_connectivity <- function(tree, undirected = FALSE, master = FALSE) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  out <- NULL

  if (undirected) {
    part_a <- tree$edge - 1
    part_b <- cbind(part_a[, 2], part_a[, 1])
    part_ab <- rbind(part_a, part_b)
    out <- part_ab
  } else {
    out <- tree$edge - 1
  }

  if (master && undirected) {
    # Nnode + 1 is the number of tips, Nnode + 2 is the index of the root node
    # Nnode + 3 is the starting index of the internal node
    # We need Nnode + 3 - 1 because of 0-based indexing in Python, thus Nnode + 2
    start_id <- tree$Nnode + 2

    # Similarly, Nnode * 2 + 1 - 1 is the ending index of the internal node
    end_id <- tree$Nnode * 2

    # Index for the new master node
    master_id <- tree$Nnode * 2 + 1

    # Add the master node to the connectivity matrix
    new_part_a <- cbind(start_id:end_id, rep(master_id, times = end_id - start_id + 1))
    new_part_b <- cbind(new_part_a[, 2], new_part_a[, 1])
    new_part_ab <- rbind(new_part_a, new_part_b)

    out <- rbind(out, new_part_ab)
  }

  if (master && !undirected) {
    stop("Master node is currently only supported for undirected trees.")
  }

  return(out)
}


#' @export rescale_crown_age
rescale_crown_age <- function(tree, target_crown_age) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  # Calculate the current crown age of the tree
  current_crown_age <- max(ape::node.depth.edgelength(tree))

  # Calculate the scaling factor
  scale_factor <- target_crown_age / current_crown_age

  # Scale the tree
  scaled_tree <- tree
  scaled_tree$edge.length <- scaled_tree$edge.length * scale_factor

  return(scaled_tree)
}


#' @export tree_to_node_feature
tree_to_node_feature <- function(tree) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  ntips <- tree$Nnode + 1

  # Assign 0 to root node, 1 to internal nodes, and 2 to tips
  node_feature <- c(rep(2, ntips), 0, rep(1, tree$Nnode - 1))

  return(node_feature)
}


#' @export tree_to_edge_feature
tree_to_edge_feature <- function(tree, undirected = FALSE) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  if (undirected) {
    return(rbind(tree$edge.length, tree$edge.length))
  } else {
    return(tree$edge.length)
  }
}


#' @export tree_to_stats
tree_to_stats <- function(tree) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  # Check if the tree is ultrametric, if not return 0
  if (!ape::is.ultrametric(tree)) {
    return(0.00)
  }

  # TODO: Maintain a named list of the stats used, ensure they are consistent
  stats <- treestats::calc_all_stats(tree)

  return(unlist(stats))
}


# TODO: Also export LTT along with branching times (I doubt this will improve the performance)
#' @export tree_to_brts
tree_to_brts <- function(tree) {
  # Check if the tree is of class 'phylo'
  if(!inherits(tree, "phylo")) {
    stop("The provided tree is not a valid phylo object.")
  }

  # Check if the tree is ultrametric, if not return 0
  if (!ape::is.ultrametric(tree)) {
    return(0.00)
  }

  brts <- sort(treestats::branching_times(tree), decreasing = TRUE)

  return(brts)
}


#' @export export_to_gps_with_params
export_to_gps_with_params <- function(data, which = "tas", undirected = FALSE) {
  path <- file.path("GPS/tree/")
  path_node <- file.path("GPS/tree/node/")
  path_edge <- file.path("GPS/tree/edge/")
  eve:::check_path(path)
  eve:::check_path(path_node)
  eve:::check_path(path_edge)

  if (which == "tas") {
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tas[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_node, "/node_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_node_feature(data$tas[[i]]), file = file_name)
    }
    for (i in seq_along(data$tas)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_edge, "/edge_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_edge_feature(data$tas[[i]], undirected = undirected), file = file_name)
    }
  } else if (which == "tes") {
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path, "/tree_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_connectivity(data$tes[[i]], undirected = undirected), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_node, "/node_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_node_feature(data$tes[[i]]), file = file_name)
    }
    for (i in seq_along(data$tes)) {
      la <- data$pars[[i]][1]
      mu <- data$pars[[i]][2]
      cap <- data$pars[[i]][3]
      age <- data$age[[i]]
      file_name <- paste0(path_edge, "/edge_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_edge_feature(data$tes[[i]]), file = file_name)
    }
  }
}
