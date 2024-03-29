#' @export mean_difference
mean_difference <- function(x, y) {
  if (length(x) != length(y)) {
    stop("Vectors must be the same length")
  }

  mean_diff <- mean(x - y)
  return(mean_diff)
}


#' @export all_differences
all_differences <- function(x, y) {
  if (length(x) != length(y)) {
    stop("Vectors must be the same length")
  }

  all_diff <- x - y
  result <- list()
  result$diff <- all_diff
  result$mle <- x
  result$true <- y
  return(result)
}


#' Generate Random Parameters Based on Specified Distributions
#'
#' This function generates random parameters based on the distribution information provided.
#' Supported distributions are Uniform, Normal, Binomial, Poisson, and Exponential.
#'
#' @param dist_info A list of lists where each inner list contains information about the
#'   distribution from which to generate random parameters. Each inner list must contain a
#'   `distribution` field specifying the distribution type (one of "uniform", "normal",
#'   "binomial", "poisson", "exponential"), an `n` field specifying the number of random
#'   values to generate, and additional fields as required by the distribution type.
#'   For example:
#'   \itemize{
#'     \item For Uniform: `min` (minimum value), `max` (maximum value)
#'     \item For Normal: `mean` (mean value), `sd` (standard deviation)
#'     \item For Binomial: `size` (number of trials), `prob` (probability of success)
#'     \item For Poisson: `lambda` (rate parameter)
#'     \item For Exponential: `rate` (rate parameter)
#'   }
#'
#' @return A list of numeric vectors, where each vector contains the generated random values
#'   from one of the specified distributions.
#'
#' @examples
#' dist_info <- list(
#'   list(distribution = "uniform", n = 10, min = 0, max = 1),
#'   list(distribution = "normal", n = 10, mean = 0, sd = 1),
#'   list(distribution = "binomial", n = 10, size = 1, prob = 0.5),
#'   list(distribution = "poisson", n = 10, lambda = 4),
#'   list(distribution = "exponential", n = 10, rate = 0.2)
#' )
#' random_params <- generate_params(dist_info)
#'
#' @export generate_params
generate_params <- function(dist_info) {
  params <- list()

  for(i in seq_along(dist_info)) {
    info <- dist_info[[i]]

    # Validate common fields
    if(!"distribution" %in% names(info) || !"n" %in% names(info)) {
      stop("Missing required fields 'distribution' or 'n' in distribution info.")
    }

    distribution <- info$distribution
    n <- info$n  # Number of random values to generate

    # Validate and generate random values based on distribution type
    if (distribution == "uniform") {
      if(!all(c("min", "max") %in% names(info))) {
        stop("Missing required fields 'min' or 'max' for uniform distribution.")
      }
      min <- info$min
      max <- info$max
      params[[i]] <- runif(n, min, max)
    } else if (distribution == "normal") {
      if(!all(c("mean", "sd") %in% names(info))) {
        stop("Missing required fields 'mean' or 'sd' for normal distribution.")
      }
      mean <- info$mean
      sd <- info$sd
      params[[i]] <- rnorm(n, mean, sd)
    } else if (distribution == "binomial") {
      if(!all(c("size", "prob") %in% names(info))) {
        stop("Missing required fields 'size' or 'prob' for binomial distribution.")
      }
      size <- info$size
      prob <- info$prob
      params[[i]] <- rbinom(n, size, prob)
    } else if (distribution == "poisson") {
      if(!"lambda" %in% names(info)) {
        stop("Missing required field 'lambda' for poisson distribution.")
      }
      lambda <- info$lambda
      params[[i]] <- rpois(n, lambda)
    } else if (distribution == "exponential") {
      if(!"rate" %in% names(info)) {
        stop("Missing required field 'rate' for exponential distribution.")
      }
      rate <- info$rate
      params[[i]] <- rexp(n, rate)
    } else if (distribution == "log10") {
      if(!all(c("min", "max") %in% names(info))) {
        stop("Missing required fields 'min' or 'max' for log10 scale uniform distribution.")
      }
      min <- info$min
      max <- info$max
      log_min <- log10(min)
      log_max <- log10(max)
      sampled_value <- runif(n, log_min, log_max)
      params[[i]] <- 10^sampled_value
    } else {
      stop(paste("Unknown distribution:", distribution))
    }
  }

  return(params)
}


#' Compute Expected Means of Distributions
#'
#' This function computes the expected mean for a list of distributions based on the
#' distribution information provided. Supported distributions are Uniform, Normal,
#' Binomial, Poisson, and Exponential.
#'
#' @param dist_info A list of lists where each inner list contains information about the
#'   distribution for which to compute the expected mean. Each inner list must contain a
#'   `distribution` field specifying the distribution type (one of "uniform", "normal",
#'   "binomial", "poisson", "exponential"), and additional fields as required by the
#'   distribution type. For example:
#'   \itemize{
#'     \item For Uniform: `min` (minimum value), `max` (maximum value)
#'     \item For Normal: `mean` (mean value)
#'     \item For Binomial: `size` (number of trials), `prob` (probability of success)
#'     \item For Poisson: `lambda` (rate parameter)
#'     \item For Exponential: `rate` (rate parameter)
#'   }
#'
#' @return A numeric vector containing the expected mean of each distribution in `dist_info`.
#'
#' @examples
#' dist_info <- list(
#'   list(distribution = "uniform", n = 10, min = 0, max = 1),
#'   list(distribution = "normal", n = 10, mean = 0, sd = 1),
#'   list(distribution = "binomial", n = 10, size = 1, prob = 0.5),
#'   list(distribution = "poisson", n = 10, lambda = 4),
#'   list(distribution = "exponential", n = 10, rate = 0.2)
#' )
#' expected_means <- compute_expected_mean(dist_info)
#' print(expected_means)
#'
#' @export compute_expected_mean
compute_expected_mean <- function(dist_info) {
  expected_means <- numeric(length(dist_info))  # Initialize a numeric vector to store expected means

  for(i in seq_along(dist_info)) {
    info <- dist_info[[i]]
    distribution <- info$distribution

    if(distribution == "uniform") {
      if(!all(c("min", "max") %in% names(info))) {
        stop("Missing required fields 'min' or 'max' for uniform distribution.")
      }
      expected_means[i] <- (info$min + info$max) / 2

    } else if(distribution == "normal") {
      if(!"mean" %in% names(info)) {
        stop("Missing required field 'mean' for normal distribution.")
      }
      expected_means[i] <- info$mean

    } else if(distribution == "binomial") {
      if(!all(c("size", "prob") %in% names(info))) {
        stop("Missing required fields 'size' or 'prob' for binomial distribution.")
      }
      expected_means[i] <- info$size * info$prob

    } else if(distribution == "poisson") {
      if(!"lambda" %in% names(info)) {
        stop("Missing required field 'lambda' for poisson distribution.")
      }
      expected_means[i] <- info$lambda

    } else if(distribution == "exponential") {
      if(!"rate" %in% names(info)) {
        stop("Missing required field 'rate' for exponential distribution.")
      }
      expected_means[i] <- 1 / info$rate

    } else {
      stop(paste("Unknown distribution:", distribution))
    }
  }

  return(expected_means)
}


#' Takes samples in the usual manner
#'
#' The standard sample function in R samples from n numbers when x = n. This is
#' unwanted behavior when the size of the vector to sample from changes
#' dynamically. This is corrected in sample2
#'
#'
#' @param x A vector of one or more elements
#' @param size A non-negative integer giving the number of items to choose.
#' @param replace Should sampling be with replacement?
#' @param prob A vector of probability weights for obtaining the elements of
#' the vector being sampled.
#' @return \item{sam}{A vector of length \code{size} that is sampled from
#' \code{x}. }
#' @author Rampal S. Etienne
#' @keywords models
#' @examples
#'
#' sample(x = 10,size = 5,replace = TRUE)
#' sample2(x = 10,size = 5,replace = TRUE)
#'
#' @export sample2
sample2 <- function(x,size,replace = FALSE,prob = NULL)
{
  if(length(x) == 1)
  {
    x <- c(x,x)
    prob <- c(prob,prob)
    if(is.null(size))
    {
      size <- 1
    }
    if(replace == FALSE & size > 1)
    {
      stop('It is not possible to sample without replacement multiple times from a single item.')
    }
  }
  sam <- sample(x,size,replace,prob)
  return(sam)
}


#' @export extract_by_range
extract_by_range <- function(tree_list, ranges) {
  # Initialize empty lists to store the results
  within_range <- list()
  outside_range <- list()

  # Iterate over each element in the list
  for (i in seq_along(tree_list)) {
    if (!is.null(tree_list[[i]]$pars) && length(tree_list[[i]]$pars) == length(ranges)) {
      # Assume initially that the element is within the range
      is_within_range <- TRUE

      # Check each par against its corresponding range
      for (j in seq_along(tree_list[[i]]$pars)) {
        if (tree_list[[i]]$pars[[j]] < ranges[[j]][1] || tree_list[[i]]$pars[[j]] > ranges[[j]][2]) {
          is_within_range <- FALSE
          break
        }
      }

      # Add to the appropriate list based on whether it is within the range or not
      if (is_within_range) {
        within_range[[length(within_range) + 1]] <- tree_list[[i]]
      } else {
        outside_range[[length(outside_range) + 1]] <- tree_list[[i]]
      }
    } else {
      # Add to outside range if number of pars doesn't match the number of ranges
      outside_range[[length(outside_range) + 1]] <- tree_list[[i]]
    }
  }

  # Return both results
  list(within_range = purrr::transpose(within_range), outside_range = purrr::transpose(outside_range))
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