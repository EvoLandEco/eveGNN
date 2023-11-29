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
  params <- list()
  cap <- c()
  ntip <- 0
  result <- list()

  while (ntip < 10) {
    params <- generate_params(dists)
    cap <- sample(cap_range[1]:cap_range[2], 1)
    result <- dd_sim(c(unlist(params), cap), age = age, ddmodel = model)
    ntip <- result$tes$Nnode + 1
  }

  return(result)
}


#' @export randomized_bd_fixed_age
randomized_bd_fixed_age <- function(dists, age) {
  params <- generate_params(dists)
  tas <- ape::rlineage(birth = params[[1]], death = params[[2]], Tmax = age)
  tes <- ape::drop.fossil(tas)
  result <- list()
  result$tas <- tas
  result$tes <- tes
  result$brts <- treestats::branching_times(tes)
  result$age <- age
  result$model <- "BD"
  result$pars <- params

  return(result)
}


#' @export randomized_pbd_fixed_age
randomized_pbd_fixed_age <- function(dists, age) {
  params <- generate_params(dists)
  result <- pbd_sim(pars = unlist(params), age = age, soc = 2)

  return(result)
}


#' @export randomized_eve_fixed_age
randomized_eve_fixed_age <- function(dists, age, model, metric, offset) {
  ntip <- 0
  params <- list()
  result <- list()

  while (ntip < 10) {
    params <- generate_params(dists)
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
