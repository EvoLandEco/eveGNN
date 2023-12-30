#' @export correct_bias_coef
correct_bias_coef <- function(data, task = NULL) {
  pars_list <- NULL
  diff_list <- NULL
  if (task == "BD") {
    pars_list <- c("lambda", "mu")
    diff_list <- c("lambda_pred", "mu_pred")
  } else if (task == "DDD") {
    pars_list <- c("lambda", "mu", "cap")
    diff_list <- c("lambda_pred", "mu_pred", "cap_pred")
  } else if (task == "PBD") {
    pars_list <- c("lambda1","lambda2", "lambda3", "mu1", "mu2")
    diff_list <- c("lambda1_pred", "lambda2_pred", "lambda3_pred", "mu1_pred", "mu2_pred")
  } else {
    stop("Invalid task")
  }

  coef_list <- list()
  for (i in 1:length(pars_list)) {
      coef_list[[i]] <- lm((data[[pars_list[i]]] - data[[diff_list[i]]]) ~ data[[pars_list[i]]])$coefficients
  }
  names(coef_list) <- pars_list

  return(coef_list)
}