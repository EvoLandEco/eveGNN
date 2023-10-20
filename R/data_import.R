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