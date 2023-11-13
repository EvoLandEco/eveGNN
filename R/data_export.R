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


#' @export export_to_gnn_with_params
export_to_gnn_with_params <- function(data, which = "tas", undirected = FALSE) {
  path <- file.path("GNN/tree/")
  path_EL <- file.path("GNN/tree/EL/")
  eve:::check_path(path)
  eve:::check_path(path_EL)

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
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tas[[i]]), file = file_name)
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
      file_name <- paste0(path_EL, "/EL_", la, "_", mu, "_", cap, "_", age, "_", i, ".rds")
      saveRDS(tree_to_adj_mat(data$tes[[i]]), file = file_name)
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
tree_to_adj_mat <- function(tree) {
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

  return(neighbor_matrix)
}


#' @export tree_to_connectivity
tree_to_connectivity <- function(tree, undirected = FALSE) {
  if (undirected) {
    part_a <- tree$edge - 1
    part_b <- cbind(part_a[, 2], part_a[, 1])
    part_ab <- rbind(part_a, part_b)
    return(part_ab)
  } else {
    return(tree$edge - 1)
  }
}
