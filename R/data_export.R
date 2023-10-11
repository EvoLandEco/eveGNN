for (i in seq_along(target_result_pd$tas)) {
  file_name <- paste0("D:/Data/GNN/tree/","pd_tas_", i, ".rds")
  saveRDS(target_result_pd$tas[[i]]$edge - 1, file = file_name)
}
for (i in seq_along(target_result_pd$tas)) {
  file_name <- paste0("D:/Data/GNN/tree/EL/","pd_tas_EL_", i, ".rds")
  saveRDS(tree_to_adj_mat(target_result_pd$tas[[i]]), file = file_name)
  print(tree_to_adj_mat(target_result_pd$tas[[i]]))
}
for (i in seq_along(target_result_ed$tas)) {
  file_name <- paste0("D:/Data/GNN/tree/","ed_tas_", i, ".rds")
  saveRDS(target_result_ed$tas[[i]]$edge - 1, file = file_name)
}
for (i in seq_along(target_result_ed$tas)) {
  file_name <- paste0("D:/Data/GNN/tree/EL/","ed_tas_EL_", i, ".rds")
  saveRDS(tree_to_adj_mat(target_result_ed$tas[[i]]), file = file_name)
}
for (i in seq_along(target_result_nnd$tas)) {
  file_name <- paste0("D:/Data/GNN/tree/","nnd_tas_", i, ".rds")
  saveRDS(target_result_nnd$tas[[i]]$edge - 1, file = file_name)
}
for (i in seq_along(target_result_nnd$tas)) {
  file_name <- paste0("D:/Data/GNN/tree/EL/","nnd_tas_EL_", i, ".rds")
  saveRDS(tree_to_adj_mat(target_result_nnd$tas[[i]]), file = file_name)
}


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
  names(all_neighbors) <- 1:(Nnode(tree) + Ntip(tree))

  return(all_neighbors)
}


tree_to_adj_mat <- function(tree) {
  neighbor_dists <- get_all_neighbors_distances(tree)

  padded_dists <- lapply(neighbor_dists, function(x) {
    if (length(x) == 1) {
      x <- c(x, 0, 0)  # Add two zeros if length is 1
    }
    if (length(x) == 2) {
      x <- c(x, 0)  # Add one zeros if length is 2
    }
    return(x)
  })

  neighbor_matrix <- do.call(rbind, padded_dists)
  colnames(neighbor_matrix) <- NULL

  return(neighbor_matrix)
}
