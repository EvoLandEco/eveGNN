#' @export randomized_ddd_fixed_age_cap
randomized_ddd_fixed_age_cap <- function(dists, cap, age, model) {
  params <- generate_params(dists)
  result <- dd_sim(c(unlist(params), cap), age = age, ddmodel = model)

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


