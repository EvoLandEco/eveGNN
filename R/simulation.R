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


