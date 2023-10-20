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