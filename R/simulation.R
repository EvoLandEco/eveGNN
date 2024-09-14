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
randomized_ddd_fixed_age <- function(dists, cap_range, max_mu, age, model) {
  ntip <- 0
  result <- list()

  while (ntip < 10) {
    params <- generate_params(dists)
    lambda <- params[[1]]
    mu <- runif(1, min = 0, max = min(max_mu, 0.9 * lambda))
    cap <- sample(cap_range[1]:cap_range[2], 1)
    result <- dd_sim(c(lambda, mu, cap), age = age, ddmodel = model)
    ntip <- result$tes$Nnode + 1
  }

  return(result)
}


#' @export fixed_ddd_fixed_age
fixed_ddd_fixed_age <- function(params, age, model) {
  ntip <- 0
  result <- list()

  while (ntip < 10) {
    lambda <- params[[1]]
    mu <- params[[2]]
    cap <- params[[3]]
    result <- dd_sim(c(lambda, mu, cap), age = age, ddmodel = model)
    ntip <- result$tes$Nnode + 1
  }

  return(result)
}


#' @export randomized_bd_fixed_age
randomized_bd_fixed_age <- function(dists, age) {
  ntip <- 0
  result <- list()

  while (ntip < 10) {
    params <- generate_params(dists)
    lambda <- params[[1]]
    mu <- runif(1, min = 0, max = 0.9 * lambda)

    # Initialize tas to NULL
    tas <- NULL
    error_occurred <- TRUE

    # Retry until no error occurs
    while (is.null(tas) || error_occurred) {
      error_occurred <- FALSE
      tas <- tryCatch({
        ape::rlineage(birth = lambda, death = mu, Tmax = age)
      }, error = function(e) {
        error_occurred <- TRUE
        NULL
      })
      tryCatch({
        ape::drop.fossil(tas)
      }, error = function(e) {
        error_occurred <- TRUE
        NULL
      })
    }

    tes <- ape::drop.fossil(tas)
    ntip <- tes$Nnode + 1
  }

  result$tas <- tas
  result$tes <- tes
  result$brts <- treestats::branching_times(tes)
  result$age <- age
  result$model <- "BD"
  params <- c(lambda, mu)
  result$pars <- params

  return(result)
}


#' @export bd_fixed_age
bd_fixed_age <- function(lambda, mu, age) {
  tas <- ape::rlineage(birth = lambda, death = mu, Tmax = age)
  tes <- ape::drop.fossil(tas)
  result <- list()
  result$tas <- tas
  result$tes <- tes
  result$brts <- treestats::branching_times(tes)
  result$age <- age
  result$model <- "BD"
  params <- c(lambda, mu)
  result$pars <- params

  return(result)
}


#' @export randomized_pbd_fixed_age
randomized_pbd_fixed_age <- function(dists, max_mus, age) {
  ntip <- 0
  result <- list()

  while (ntip < 10) {
    params <- generate_params(dists)
    b1 <- params[[1]]
    la1 <- params[[2]]
    b2 <- params[[3]]
    mu1 <- runif(1, min = 0, max = min(max_mus[1], 0.8 * b1))
    mu2 <- runif(1, min = 0, max = min(max_mus[2], 0.8 * b2))
    result <- pbd_sim(pars = c(b1, la1, b2, mu1, mu2), age = age, soc = 2)
    ntip <- result$tes$Nnode + 1
  }

  return(result)
}


#' @export randomized_eve_fixed_age
randomized_eve_fixed_age <- function(dists, age, metric, offset) {
  result <- list()

  params <- generate_params(dists)
  lambda <- params[[1]]
  mu <- runif(1, min = 0, max = 0.8 * lambda)
  beta_n <- params[[2]]
  beta_phi <- params[[3]]
  gamma_n <- params[[4]]
  gamma_phi <- params[[5]]

  pars_list <- c(lambda, mu, beta_n, beta_phi, gamma_n, gamma_phi)
  print(pars_list)
  evesim::edd_sim
  raw_result <- evesim::edd_sim(pars = pars_list,
                         age = age,
                         metric = metric,
                         offset = offset,
                         size_limit = 2000)

  if (is.null(raw_result$sim)) {
    result[["tes"]] <- NULL
  } else {
    result[["tes"]] <- evesim::SimTable.phylo(raw_result$sim, drop_extinct = TRUE)
    # result[["tas"]] <- evesim::SimTable.phylo(raw_result$sim, drop_extinct = FALSE)
  }

  result[["pars"]] <- pars_list
  result[["age"]] <- age
  result[["model"]] <- "dsde2" # hard-coded for backward compatibility
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


#' @export bd_sim_fix_n
bd_sim_fix_n <- function(n, pars, age) {
  desired_data <- NULL
  while(is.null(desired_data)) {
    tryCatch({
      sim_data <- ape::rlineage(birth = pars[1], death = pars[2], Tmax = age)
      sim_data <- ape::drop.fossil(sim_data)
      if(sim_data$Nnode == (n - 1)) {
        desired_data <- sim_data
      }
    }, error = function(e) {
      # Handle error
      message("Error in bd_sim: ", e$message)
    })
  }
  return(desired_data)
}


#' @export pbd_sim_fix_n
pbd_sim_fix_n <- function(n, pars, age, soc) {
  desired_data <- NULL
  while(is.null(desired_data)) {
    tryCatch({
      sim_data <- pbd_sim(pars, age, soc = soc)
      if(sim_data$tes$Nnode == (n - 1)) {
        desired_data <- sim_data
      }
    }, error = function(e) {
      # Handle error
      message("Error in pbd_sim: ", e$message)
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


#' @export tree_polymorphism_bootstrap
tree_polymorphism_bootstrap <- function(pars, age, ntip, model, nrep = 1000) {
  result <- list()

  while (length(result) < nrep) {
    sim_data <- NULL
    if (model == "BD") {
      sim_data <- bd_sim_fix_n(n = ntip, pars = pars, age = age)
    } else if (model == "DDD") {
      sim_data <- dd_sim_fix_n(n = ntip, pars = pars, age = age, ddmodel = 1)
      sim_data <- sim_data$tes
    } else if (model == "PBD") {
      sim_data <- pbd_sim_fix_n(n = ntip, pars = pars, age = age, soc = 2)
      sim_data <- sim_data$tes
    }

    if (!is.null(sim_data)) {
      result[[length(result) + 1]] <- sim_data
    }
  }

  return(result)
}