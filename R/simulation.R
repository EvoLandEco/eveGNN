#' @export randomized_ddd_fixed_age_cap
randomized_ddd_fixed_age_cap <- function(dists, cap, age, model) {
  params <- generate_params(dists)
  result <- dd_sim(c(unlist(params), cap), age = age, ddmodel = model)

  return(result)
}


#' @export randomized_ddd_fixed_la_mu_age
randomized_ddd_fixed_la_mu_age <- function(cap_range, la, mu, age, model) {
  cap <- sample(cap_range[1]:cap_range[2], 1)
  result <- dd_sim(c(la, mu, cap), age = age, ddmodel = model)

  return(result)
}


#' @export randomized_ddd_fixed_age
randomized_ddd_fixed_age <- function(dists, cap_range, age, model) {
  params <- generate_params(dists)
  cap <- sample(cap_range[1]:cap_range[2], 1)

  ntip <- 0
  result <- list()
  while (ntip < 10) {
    result <- dd_sim(c(unlist(params), cap), age = age, ddmodel = model)
    ntip <- result$tes$Nnode + 1
  }

  return(result)
}


#' @export randomized_eve_fixed_age
randomized_eve_fixed_age <- function(dists, age, model, metric, offset) {
  params <- generate_params(dists)

  ntip <- 0
  result <- list()
  while (ntip < 10) {
    result <- eve::edd_sim(unlist(params), age = age, model = model, metric = metric, offset = offset, history = FALSE)
    ntip <- result$tes$Nnode + 1
  }

  result[["pars"]] <- unlist(params)
  result[["age"]] <- age
  result[["model"]] <- model
  result[["metric"]] <- metric
  result[["offset"]] <- offset

  return(result)
}


#' @export batch_sim_ddd
batch_sim_ddd <- function(dists, cap, age, model, nrep = 10) {
  batch <- list()
  for (i in 1:nrep) {
    batch[[i]] <- randomized_ddd_fixed_age_cap(dists = dists, cap = cap, age = age, model = model)
  }
  trans_batch <- purrr::transpose(batch)

  return(trans_batch)
}


#' @export dd_sim_fix_n
dd_sim_fix_n <- function(n, pars, age, ddmodel = 1) {
  desired_data <- NULL
  while(is.null(desired_data)) {
    tryCatch({
      sim_data <- dd_sim(pars, age, ddmodel = ddmodel)
      if(sim_data$tes$Nnode == (n - 1)) {
        desired_data <- sim_data
      }
    }, error = function(e) {
      # Handle error
      message("Error in dd_sim: ", e$message)
    })
  }
  return(desired_data)
}


#' @export edd_sim_fix_n
edd_sim_fix_n <- function(n, pars, age, model, metric, offset) {
  desired_data <- NULL
  while(is.null(desired_data)) {
    tryCatch({
      sim_data <- eve::edd_sim(pars, age, model = model, metric = metric, offset = offset, history = FALSE)
      if(sim_data$tes$Nnode == (n - 1)) {
        desired_data <- sim_data
      }
    }, error = function(e) {
      # Handle error
      message("Error in edd_sim: ", e$message)
    })
  }
  return(desired_data)
}


#' @export get_test_data
get_test_data <- function(original_list, quantile) {
  # Ensure the quantile argument is within a valid range
  if (quantile <= 0 || quantile > 1) {
    stop("Quantile must be between 0 and 1 (exclusive).")
  }

  # Initialize an empty list to store the extracted data
  test_data_list <- list()

  # Iterate through each sublist in the original list
  for (sublist_name in names(original_list)) {
    # Retrieve the current sublist
    sublist <- original_list[[sublist_name]]

    # Determine the number of elements to extract based on the quantile argument
    num_sublists <- length(sublist)
    num_to_extract <- floor(quantile * num_sublists)

    # Extract the specified proportion of data from the end of the sublist
    test_data <- tail(sublist, n = num_to_extract)

    # Store the extracted data in the test_data_list using the sublist_name as the list name
    test_data_list[[sublist_name]] <- test_data
  }

  return(test_data_list)
}