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
  for (file_name in all_files) {
    data <- readRDS(file.path(path, "umap", file_name))

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

  return(dataset)
}