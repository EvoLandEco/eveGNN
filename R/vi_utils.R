grouping_var_to_label <- function(grouping_var) {
  if (grouping_var == "cap") {
    return("Carrying capacity")
  } else if (grouping_var == "beta_n") {
    return("beta_n")
  } else if (grouping_var == "beta_phi") {
    return("beta_phi")
  } else {
    stop("Invalid grouping variable")
  }
}


difference_var_to_label <- function(difference_var) {
  difference_var = as.character(difference_var)
  for (i in 1:length(difference_var)) {
    if (difference_var[i] == "lambda_r_diff") {
      difference_var[i] <- "lambda"
    } else if (difference_var[i] == "mu_r_diff") {
      difference_var[i] <- "mu"
    } else if (difference_var[i] == "cap_r_diff") {
      difference_var[i] <- "K"
    } else {
      stop("Invalid difference variable")
    }
  }
  return(difference_var)
}