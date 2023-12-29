grouping_var_to_label <- function(grouping_var) {
  if (grouping_var == "lambda") {
    return("Speciation rate")
  } else if (grouping_var == "mu") {
    return("Extinction rate")
  } else if (grouping_var == "cap") {
    return("Carrying capacity")
  } else if (grouping_var == "beta_n") {
    return("Species richness effect")
  } else if (grouping_var == "beta_phi") {
    return("Evolutionary relatedness effect")
  } else if (grouping_var == "lambda1") {
    return("Speciation rate of good species")
  } else if (grouping_var == "lambda2") {
    return("Speciation-completion rate")
  } else if (grouping_var == "lambda3") {
    return("Speciation rate of incipient species")
  } else if (grouping_var == "mu1") {
    return("Extinction rate of good species")
  } else if (grouping_var == "mu2") {
    return("Extinction rate of incipient species")
  } else if (grouping_var == "(lambda - mu)") {
    return("Net speciation rate")
  } else if (grouping_var == "num_nodes") {
    return("Number of nodes")
  } else {
    stop("Invalid grouping variable")
  }
}


difference_var_to_label <- function(difference_var) {
  difference_var <-  as.character(difference_var)
  for (i in 1:length(difference_var)) {
    if (difference_var[i] == "lambda_r_diff" || difference_var[i] == "lambda_a_diff") {
      difference_var[i] <- "lambda[0]"
    } else if (difference_var[i] == "mu_r_diff" || difference_var[i] == "mu_a_diff") {
      difference_var[i] <- "mu[0]"
    } else if (difference_var[i] == "cap_r_diff" || difference_var[i] == "cap_a_diff") {
      difference_var[i] <- "K"
    } else if (difference_var[i] == "beta_n_r_diff" || difference_var[i] == "beta_n_a_diff") {
      difference_var[i] <- "beta[italic(N)]"
    } else if (difference_var[i] == "beta_phi_r_diff" || difference_var[i] == "beta_phi_a_diff") {
      difference_var[i] <- "beta[italic(Phi)]"
    } else if (difference_var[i] == "lambda1_r_diff" || difference_var[i] == "lambda1_a_diff") {
      difference_var[i] <- "b[1]"
    } else if (difference_var[i] == "lambda2_r_diff" || difference_var[i] == "lambda2_a_diff") {
      difference_var[i] <- "lambda[1]"
    } else if (difference_var[i] == "lambda3_r_diff" || difference_var[i] == "lambda3_a_diff") {
      difference_var[i] <- "b[2]"
    } else if (difference_var[i] == "mu1_r_diff" || difference_var[i] == "mu1_a_diff") {
      difference_var[i] <- "mu[1]"
    } else if (difference_var[i] == "mu2_r_diff" || difference_var[i] == "mu2_a_diff") {
      difference_var[i] <- "mu[2]"
    } else {
      stop("Invalid difference variable")
    }
  }
  return(difference_var)
}


vir_lite <- function(cols, ds=0.4, dv=0.7) {
  cols <- rgb2hsv(col2rgb(cols))
  cols["v", ] <- cols["v", ] + dv*(1 - cols["v", ])
  cols["s", ] <- ds*cols["s", ]
  apply(cols, 2, function(x) hsv(x[1], x[2], x[3]))
}