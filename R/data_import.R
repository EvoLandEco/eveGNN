#' @export parse_filename
parse_filename <- function(filename) {
  filename <- str_remove(filename, "\\.rds")
  # Split the filename into parts separated by underscores
  parts <- str_split(filename, "_")[[1]]

  # Determine the group (EVE or DDD)
  group <- parts[1]

  # Initialize an empty list to store the extracted values
  values <- list()

  if (group == "EVE") {
    # Parse the values for EVE group
    values$set <- as.integer(parts[4])
    values$beta_n <- as.numeric(parts[5])
    values$beta_phi <- as.numeric(parts[6])
    values$metric <- parts[7]
    values$lambda <- as.numeric(parts[8])
    values$mu <- as.numeric(parts[9])
    values$age <- as.numeric(parts[10])
    values$n <- as.integer(parts[11])
    values$type <- parts[1]
  } else if (group == "DDD") {
    # Parse the values for DDD group
    values$set <- as.integer(parts[4])
    values$cap <- as.integer(parts[5])
    values$lambda <- as.numeric(parts[6])
    values$mu <- as.numeric(parts[7])
    values$age <- as.numeric(parts[8])
    values$n <- as.integer(parts[9])
    values$type <- parts[1]
  } else {
    stop("Unknown group: ", group)
  }

  return(values)
}


#' @export load_model_result
load_model_result <- function(path) {
  rds_files <- list.files(path = path, pattern = "\\.rds$", full.names = FALSE)

  data_list <- list()
  for (i in seq_along(rds_files)) {
    params <- parse_filename(rds_files[i])
    data <- readRDS(file.path(path, rds_files[i]))
    data_list[[i]] <- list(data = data, params = params)
  }

  return(data_list)
}


#' @export group_by_type
group_by_type <- function(input_list) {
  # Initialize an empty list to store the output
  output_list <- list()

  # Iterate through each sublist in the input list
  for (i in seq_along(input_list)) {
    # Extract the type value from the params sublist
    type_value <- input_list[[i]]$params$type

    # If this type hasn't been seen before, initialize a new sublist for it
    if (!is.null(type_value) && !type_value %in% names(output_list)) {
      output_list[[type_value]] <- list()
    }

    # Append the current sublist to the appropriate type sublist in the output list
    if (!is.null(type_value)) {
      output_list[[type_value]] <- c(output_list[[type_value]], list(input_list[[i]]))
    }
  }

  return(output_list)
}


#' @export joint_by_type
joint_by_type <- function(input_list) {
  # Iterate through each sublist in the input list
  for (i in seq_along(input_list)) {
    data_list <- list()
    for (j in seq_along(input_list[[i]])) {
      data <- input_list[[i]][[j]]$data
      params_df <- as.data.frame(t(unlist(input_list[[i]][[j]]$params)))
      params_df <- params_df[rep(1:nrow(params_df), each = nrow(data)), , drop = FALSE]
      data <- cbind(data, params_df)
      data_list[[j]] <- data
    }
    input_list[[i]] <- do.call(rbind, data_list)
  }

  return(input_list)
}


#' @export read_umap_data
read_umap_data <- function(path) {
  # List all .rds files recursively within the umap_folder
  all_files <- list.files(path = file.path(path, "umap"), pattern = "\\.rds$", recursive = TRUE, full.names = FALSE)

  # Initialize an empty list to store the datasets
  dataset <- list()
  i <- 1
  # Loop through each file and read the .rds file and parse metadata
  for (file_path in all_files) {
    data <- readRDS(file.path(path, "umap", file_path))
    # Parse metadata from the filename and path
    path_parts <- unlist(strsplit(file_path, split = "/"))
    file_name <- tail(path_parts, n=1) # Get the last part (filename) from the path
    sub_folder <- tail(path_parts, n=2)[1] # Get the second last part (subfolder) from the path

    # Extract metadata components
    sub_folder_parts <- unlist(strsplit(sub_folder, split = "_"))
    model <- sub_folder_parts[1]
    which <- sub_folder_parts[2]

    file_name_parts <- unlist(strsplit(file_name, split = "_"))
    set <- as.numeric(gsub("set", "", file_name_parts[2]))
    type <- file_name_parts[3]
    stat <- file_name_parts[4]
    epoch <- as.numeric(gsub("epoch", "", gsub("\\.rds", "", file_name_parts[6])))

    dataset[[i]] <- list(
      data = data,
      model = model,
      which = which,
      set = set,
      type = type,
      stat = stat,
      epoch = epoch
    )

    i <- i + 1
  }

  dataset <- purrr::transpose(dataset)

  return(dataset)
}


#' @export compute_umap
compute_umap <- function(dataset) {
  dataset$data <- lapply(dataset$data, function(x) {
    x[-ncol(x)] <- lapply(x[-ncol(x)], as.numeric)
    x_data <- x[-ncol(x)]
    x_data <- umap(x_data)
    x_plot_data <- cbind(as.data.frame(x_data$layout), label = x$label)

    return(x_plot_data)
  })

  return(dataset)
}


#' @export extract_umap_data_by_epoch
extract_umap_data_by_epoch <- function(dataset, target_epochs = c(1, 50, 199)) {
  # Check the validity of target_epochs
  if (!all(is.numeric(target_epochs))) {
    stop("All target epochs must be numeric values.")
  }

  # Check if 'epoch' sublist exists within the dataset
  if (!"epoch" %in% names(dataset)) {
    stop("The dataset must have 'epoch' as a sublist.")
  }

  # Find the indices where epoch is in target_epochs
  selected_indices <- which(sapply(dataset$epoch, function(e) e %in% target_epochs))

  # Extract elements based on the indices from all sublists
  results <- lapply(names(dataset), function(name) {
    dataset[[name]][selected_indices]
  })

  # Convert the results to a named list for clarity
  names(results) <- names(dataset)

  return(results)
}


#' @export extract_umap_data_by_metadata
extract_umap_data_by_metadata <- function(dataset,
                                          target_model = NULL,
                                          target_which = NULL,
                                          target_set = NULL,
                                          target_type = NULL,
                                          target_stat = NULL) {

  # Check if the required sublists exist within the dataset
  required_sublists <- c("model", "which", "set", "type", "stat", "data")
  if (!all(required_sublists %in% names(dataset))) {
    stop("The dataset must have 'model', 'which', 'set', 'type', 'stat', and 'data' as sublists.")
  }

  # Setting default values for the logical checks to TRUE
  selected_logic <- rep(TRUE, length(dataset$data))

  # Update the logical checks based on provided criteria
  if (!is.null(target_model)) selected_logic <- selected_logic & dataset$model %in% target_model
  if (!is.null(target_which)) selected_logic <- selected_logic & dataset$which %in% target_which
  if (!is.null(target_set)) selected_logic <- selected_logic & dataset$set %in% target_set
  if (!is.null(target_type)) selected_logic <- selected_logic & dataset$type %in% target_type
  if (!is.null(target_stat)) selected_logic <- selected_logic & dataset$stat %in% target_stat

  # Identify the selected indices
  selected_indices <- which(selected_logic)

  # Extract elements based on the indices from all sublists
  results <- lapply(names(dataset), function(name) {
    dataset[[name]][selected_indices]
  })

  # Convert the results to a named list for clarity
  names(results) <- names(dataset)

  return(results)
}


#' @export find_unique_combinations
find_unique_combinations <- function(dataset, ...) {

  # Convert ... to a list
  targets <- as.character(substitute(list(...)))[-1]

  # Extract values for each target across the dataset
  target_values <- lapply(targets, function(target) {
    sapply(dataset[[target]], function(sublist_val) sublist_val)
  })

  # Convert the list of values into a dataframe
  df <- as.data.frame(target_values)
  colnames(df) <- targets

  # Find unique combinations
  unique_combinations <- unique(df)
  rownames(unique_combinations) <- NULL

  return(unique_combinations)
}


load_performance_data <- function(path, task_type, model_type, max_depth) {
  performances <- list()

  # construct filenames
  for (depth in seq_len(max_depth)) {
    file_name <- paste0(task_type, "_", model_type, "_", depth, ".rds")
    file_path <- file.path(path, task_type, file_name)
    performances[[length(performances) + 1]] <- readRDS(file_path)
  }

  for (i in seq_len(max_depth)) {
    performances[[i]]$Depth <- i
    performances[[i]]$Model <- model_type
    performances[[i]]$Task <- task_type
  }

  performances <- dplyr::bind_rows(performances)
  rownames(performances) <- NULL
  performances$Epoch <- as.integer(performances$Epoch)
  performances$Depth <- as.integer(performances$Depth)
  performances <- as.data.frame(performances)

  return(performances)
}


load_final_difference <- function(path, task_type, model_type, max_depth) {
  final_differences <- list()
  final_predictions <- list()
  final_true_values <- list()
  differences <- list()
  # construct filenames
  for (depth in seq_len(max_depth)) {
      file_name <- paste0(task_type, "_", "final_diffs", "_", model_type, "_", depth, ".rds")
      file_path <- file.path(path, task_type, file_name)
      final_differences[[length(final_differences) + 1]] <- readRDS(file_path)
  }
  for (depth in seq_len(max_depth)) {
    file_name <- paste0(task_type, "_", "final_predictions", "_", model_type, "_", depth, ".rds")
    file_path <- file.path(path, task_type, file_name)
    final_predictions[[length(final_predictions) + 1]] <- readRDS(file_path)
  }
  for (depth in seq_len(max_depth)) {
    file_name <- paste0(task_type, "_", "final_y", "_", model_type, "_", depth, ".rds")
    file_path <- file.path(path, task_type, file_name)
    final_true_values[[length(final_true_values) + 1]] <- readRDS(file_path)
  }

  for (i in seq_len(max_depth)) {
    differences[[i]] <- cbind(final_differences[[i]], final_predictions[[i]], final_true_values[[i]])
    differences[[i]] <- dplyr::mutate_all(differences[[i]], as.numeric)
    differences[[i]]$Depth <- i
    differences[[i]]$Model <- model_type
    differences[[i]]$Task <- task_type
    differences[[i]] <- as.data.frame(differences[[i]])

    if (task_type != "PBD_FREE_TES" && task_type != "PBD_FREE_TAS" && task_type != "PBD_VAL_TES" && task_type != "PBD_VAL_TAS") {
      # compute lambda relative difference
      differences[[i]]$lambda_r_diff <- (differences[[i]]$lambda - differences[[i]]$lambda_pred) / differences[[i]]$lambda * 100
      # compute lambda absolute difference
      differences[[i]]$lambda_a_diff <- differences[[i]]$lambda - differences[[i]]$lambda_pred
      # compute mu relative difference
      differences[[i]]$mu_r_diff <- (differences[[i]]$mu - differences[[i]]$mu_pred) / differences[[i]]$mu * 100
      # compute mu absolute difference
      differences[[i]]$mu_a_diff <- differences[[i]]$mu - differences[[i]]$mu_pred
    }

    if (task_type == "DDD_FREE_TES" || task_type == "DDD_FREE_TAS" || task_type == "DDD_VAL_TES" || task_type == "DDD_VAL_TAS") {
      # multiply cap by 1000
      differences[[i]]$cap <- differences[[i]]$cap * 1000
      # multiply cap_pred by 1000
      differences[[i]]$cap_pred <- differences[[i]]$cap_pred * 1000
      # compute cap relative difference
      differences[[i]]$cap_r_diff <- (differences[[i]]$cap - differences[[i]]$cap_pred) / differences[[i]]$cap * 100
      # compute cap absolute difference
      differences[[i]]$cap_a_diff <- differences[[i]]$cap - differences[[i]]$cap_pred
    }

    if (task_type == "EVE_FREE_TES" || task_type == "EVE_FREE_TAS" || task_type == "EVE_VAL_TES" || task_type == "EVE_VAL_TAS") {
      # compute beta_n relative difference
      differences[[i]]$beta_n_r_diff <- (differences[[i]]$beta_n - differences[[i]]$beta_n_pred) / differences[[i]]$beta_n * 100
      # compute beta_n absolute difference
      differences[[i]]$beta_n_a_diff <- differences[[i]]$beta_n - differences[[i]]$beta_n_pred
      # compute beta_phi relative difference
      differences[[i]]$beta_phi_r_diff <- (differences[[i]]$beta_phi - differences[[i]]$beta_phi_pred) / differences[[i]]$beta_phi * 100
      # compute beta_phi absolute difference
      differences[[i]]$beta_phi_a_diff <- differences[[i]]$beta_phi - differences[[i]]$beta_phi_pred
    }

    if (task_type == "PBD_FREE_TES" || task_type == "PBD_FREE_TAS" || task_type == "PBD_VAL_TES" || task_type == "PBD_VAL_TAS") {
      # compute lambda1 relative difference
      differences[[i]]$lambda1_r_diff <- (differences[[i]]$lambda1 - differences[[i]]$lambda1_pred) / differences[[i]]$lambda1 * 100
      # compute lambda1 absolute difference
      differences[[i]]$lambda1_a_diff <- differences[[i]]$lambda1 - differences[[i]]$lambda1_pred
      # compute lambda2 relative difference
      differences[[i]]$lambda2_r_diff <- (differences[[i]]$lambda2 - differences[[i]]$lambda2_pred) / differences[[i]]$lambda2 * 100
      # compute lambda2 absolute difference
      differences[[i]]$lambda2_a_diff <- differences[[i]]$lambda2 - differences[[i]]$lambda2_pred
      # compute lambda3 relative difference
      differences[[i]]$lambda3_r_diff <- (differences[[i]]$lambda3 - differences[[i]]$lambda3_pred) / differences[[i]]$lambda3 * 100
      # compute lambda3 absolute difference
      differences[[i]]$lambda3_a_diff <- differences[[i]]$lambda3 - differences[[i]]$lambda3_pred
      # compute mu1 relative difference
      differences[[i]]$mu1_r_diff <- (differences[[i]]$mu1 - differences[[i]]$mu1_pred) / differences[[i]]$mu1 * 100
      # compute mu1 absolute difference
      differences[[i]]$mu1_a_diff <- differences[[i]]$mu1 - differences[[i]]$mu1_pred
      # compute mu2 relative difference
      differences[[i]]$mu2_r_diff <- (differences[[i]]$mu2 - differences[[i]]$mu2_pred) / differences[[i]]$mu2 * 100
      # compute mu2 absolute difference
      differences[[i]]$mu2_a_diff <- differences[[i]]$mu2 - differences[[i]]$mu2_pred
    }
  }

  differences <- dplyr::bind_rows(differences)
  rownames(differences) <- NULL

  return(differences)
}


load_final_difference_by_layer <- function(path, task_type, model_type, depth) {
  # construct filenames
  file_name <- paste0(task_type, "_", "final_diffs", "_", model_type, "_", depth, ".rds")
  file_path <- file.path(path, task_type, file_name)
  final_difference <- readRDS(file_path)

  file_name <- paste0(task_type, "_", "final_predictions", "_", model_type, "_", depth, ".rds")
  file_path <- file.path(path, task_type, file_name)
  final_prediction <- readRDS(file_path)


  file_name <- paste0(task_type, "_", "final_y", "_", model_type, "_", depth, ".rds")
  file_path <- file.path(path, task_type, file_name)
  final_true_value <- readRDS(file_path)

  differences <- cbind(final_difference, final_prediction, final_true_value)
  differences <- dplyr::mutate_all(differences, as.numeric)
  differences$Depth <- depth
  differences$Model <- model_type
  differences$Task <- task_type
  differences <- as.data.frame(differences)

  if (task_type != "PBD_FREE_TES" && task_type != "PBD_FREE_TAS" && task_type != "PBD_VAL_TES" && task_type != "PBD_VAL_TAS") {
    # compute lambda relative difference
    differences$lambda_r_diff <- (differences$lambda - differences$lambda_pred) / differences$lambda * 100
    # compute lambda absolute difference
    differences$lambda_a_diff <- differences$lambda - differences$lambda_pred
    # compute mu relative difference
    differences$mu_r_diff <- (differences$mu - differences$mu_pred) / differences$mu * 100
    # compute mu absolute difference
    differences$mu_a_diff <- differences$mu - differences$mu_pred
  }

  if (task_type == "DDD_FREE_TES" || task_type == "DDD_FREE_TAS" || task_type == "DDD_VAL_TES" || task_type == "DDD_VAL_TAS") {
    # multiply cap by 1000
    differences$cap <- differences$cap * 1000
    # multiply cap_pred by 1000
    differences$cap_pred <- differences$cap_pred * 1000
    # compute cap relative difference
    differences$cap_r_diff <- (differences$cap - differences$cap_pred) / differences$cap * 100
    # compute cap absolute difference
    differences$cap_a_diff <- differences$cap - differences$cap_pred
  }

  if (task_type == "EVE_FREE_TES" || task_type == "EVE_FREE_TAS" || task_type == "EVE_VAL_TES" || task_type == "EVE_VAL_TAS") {
    # compute beta_n relative difference
    differences$beta_n_r_diff <- (differences$beta_n - differences$beta_n_pred) / differences$beta_n * 100
    # compute beta_n absolute difference
    differences$beta_n_a_diff <- differences$beta_n - differences$beta_n_pred
    # compute beta_phi relative difference
    differences$beta_phi_r_diff <- (differences$beta_phi - differences$beta_phi_pred) / differences$beta_phi * 100
    # compute beta_phi absolute difference
    differences$beta_phi_a_diff <- differences$beta_phi - differences$beta_phi_pred
  }

  if (task_type == "PBD_FREE_TES" || task_type == "PBD_FREE_TAS" || task_type == "PBD_VAL_TES" || task_type == "PBD_VAL_TAS") {
    # compute lambda1 relative difference
    differences$lambda1_r_diff <- (differences$lambda1 - differences$lambda1_pred) / differences$lambda1 * 100
    # compute lambda1 absolute difference
    differences$lambda1_a_diff <- differences$lambda1 - differences$lambda1_pred
    # compute lambda2 relative difference
    differences$lambda2_r_diff <- (differences$lambda2 - differences$lambda2_pred) / differences$lambda2 * 100
    # compute lambda2 absolute difference
    differences$lambda2_a_diff <- differences$lambda2 - differences$lambda2_pred
    # compute lambda3 relative difference
    differences$lambda3_r_diff <- (differences$lambda3 - differences$lambda3_pred) / differences$lambda3 * 100
    # compute lambda3 absolute difference
    differences$lambda3_a_diff <- differences$lambda3 - differences$lambda3_pred
    # compute mu1 relative difference
    differences$mu1_r_diff <- (differences$mu1 - differences$mu1_pred) / differences$mu1 * 100
    # compute mu1 absolute difference
    differences$mu1_a_diff <- differences$mu1 - differences$mu1_pred
    # compute mu2 relative difference
    differences$mu2_r_diff <- (differences$mu2 - differences$mu2_pred) / differences$mu2 * 100
    # compute mu2 absolute difference
    differences$mu2_a_diff <- differences$mu2 - differences$mu2_pred
  }

  rownames(differences) <- NULL

  return(differences)
}


load_full_mle_result <- function(path, task_type, model_type) {
  # construct filenames
  file_name <- paste0("mle_diffs", "_", task_type, ".rds")
  file_path <- file.path(path, task_type, file_name)
  mle_results <- readRDS(file_path)

  out <- list()
  for (i in seq_len(length(mle_results))) {
    r_diffs <- (mle_results[[i]]$true - mle_results[[i]]$mle) / mle_results[[i]]$true * 100
    out[[i]] <- list(
      lambda = mle_results[[i]]$true[1],
      mu = mle_results[[i]]$true[2],
      lambda_r_diff = r_diffs[1],
      mu_r_diff = r_diffs[2],
      num_nodes = mle_results[[i]]$nnode
    )
  }

  out <- dplyr::bind_rows(out)
  out$Model <- model_type
  out$Task <- task_type
  out <- as.data.frame(out)

  return(out)
}


#' @export load_separated_mle_result
load_separated_mle_result <- function(path, task_type, model_type) {
    # construct filenames
  mle_list <- list.files(file.path(path, paste0(task_type, "_MLE_TES")), pattern = "^differences_[0-9]+\\.rds$", full.names = TRUE)
  mle_results <- list()
  out <- list()
  for (i in seq_len(length(mle_list))) {
    mle_results[[i]] <- readRDS(mle_list[i])
  }
  if (task_type == "DDD") {
    for (i in seq_len(length(mle_results))) {
      r_diffs <- (mle_results[[i]]$true - mle_results[[i]]$mle) / mle_results[[i]]$true * 100
      a_diffs <- mle_results[[i]]$true - mle_results[[i]]$mle
      out[[i]] <- list(
        lambda = mle_results[[i]]$true[1],
        mu = mle_results[[i]]$true[2],
        cap = mle_results[[i]]$true[3],
        lambda_r_diff = r_diffs[1],
        mu_r_diff = r_diffs[2],
        cap_r_diff = r_diffs[3],
        lambda_a_diff = a_diffs[1],
        mu_a_diff = a_diffs[2],
        cap_a_diff = a_diffs[3],
        num_nodes = mle_results[[i]]$nnode
      )
    }
  } else if (task_type == "PBD") {
    for (i in seq_len(length(mle_results))) {
      r_diffs <- (mle_results[[i]]$true - mle_results[[i]]$mle) / mle_results[[i]]$true * 100
      a_diffs <- mle_results[[i]]$true - mle_results[[i]]$mle
      out[[i]] <- list(
        lambda1 = mle_results[[i]]$true[1],
        lambda2 = mle_results[[i]]$true[2],
        lambda3 = mle_results[[i]]$true[3],
        mu1 = mle_results[[i]]$true[4],
        mu2 = mle_results[[i]]$true[5],
        lambda1_r_diff = r_diffs[1],
        lambda2_r_diff = r_diffs[2],
        lambda3_r_diff = r_diffs[3],
        mu1_r_diff = r_diffs[4],
        mu2_r_diff = r_diffs[5],
        lambda1_a_diff = a_diffs[1],
        lambda2_a_diff = a_diffs[2],
        lambda3_a_diff = a_diffs[3],
        mu1_a_diff = a_diffs[4],
        mu2_a_diff = a_diffs[5],
        num_nodes = mle_results[[i]]$nnode
      )
    }
  } else if (task_type == "EVE") {
    stop("Not implemented yet.")
  } else {
    stop("Unknown task type: ", task_type)
  }

  out <- dplyr::bind_rows(out)
  out$Model <- model_type
  out$Task <- paste0(task_type, "_MLE_TES")
  out <- as.data.frame(out)

  return(out)
}