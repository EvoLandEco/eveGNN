#' @export compute_accuracy_dd_ml
compute_accuracy_dd_ml <- function(dist_info, data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  mean_pars <- eveGNN::compute_expected_mean(dist_info)
  diffs <- furrr::future_map(.x = seq_along(data$brts),
                             .f = function(i) {
                               ml <- tryCatch(
                               {
                                 DDD::dd_ML(
                                   brts = data$brts[[i]],
                                   initparsopt = mean_pars,
                                   idparsopt = c(1, 2),
                                   idparsfix = c(3),
                                   parsfix = data$pars[[i]][3],
                                   btorph = 0,
                                   soc = 2,
                                   cond = 1,
                                   ddmodel = 1,
                                   num_cycles = Inf
                                 )
                               },
                                 error = function(e) {
                                   NA
                                 }
                               )
                               # If an error occurred, ml will be NA and we return NA right away.
                               if (length(ml) == 1) {
                                 if (is.na(ml)) {
                                   return(NA)
                                 }
                               }
                               # If no error occurred, proceed as before.
                               names(ml) <- NULL
                               eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])
                             })
  return(diffs)
}


#' @export compute_accuracy_dd_ml_fix_lamu
compute_accuracy_dd_ml_fix_lamu <- function(cap_range, data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  mean_cap <- mean(cap_range)
  diffs <- furrr::future_map(.x = seq_along(data$brts),
                             .f = function(i) {
                               ml <- tryCatch(
                               {
                                 DDD::dd_ML(
                                   brts = data$brts[[i]],
                                   initparsopt = mean_cap,
                                   idparsopt = c(3),
                                   idparsfix = c(1, 2),
                                   parsfix = data$pars[[i]][1:2],
                                   btorph = 0,
                                   soc = 2,
                                   cond = 0,
                                   ddmodel = 1,
                                   num_cycles = 1
                                 )
                               },
                                 error = function(e) {
                                   NA
                                 }
                               )
                               # If an error occurred, ml will be NA and we return NA right away.
                               if (length(ml) == 1) {
                                 if (is.na(ml)) {
                                   return(NA)
                                 }
                               }
                               # If no error occurred, proceed as before.
                               names(ml) <- NULL
                               eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])
                             })
  return(diffs)
}


#' @export compute_accuracy_dd_ml_free
compute_accuracy_dd_ml_free <- function(dist_info, cap_range, data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  mean_lambda <- eveGNN::compute_expected_mean(dist_info)
  mean_mu <- mean_lambda * 0.9
  mean_cap <- mean(cap_range)
  diffs <- furrr::future_map(.x = seq_along(data$brts),
                             .f = function(i) {
                               ml <- DDD::dd_ML(
                                 brts = data$brts[[i]],
                                 initparsopt = data$pars[[i]],
                                 idparsopt = c(1, 2, 3),
                                 btorph = 0,
                                 soc = 2,
                                 cond = 1,
                                 ddmodel = 1,
                                 num_cycles = 3
                               )
                               # If an error occurred, ml will be NA and we return NA right away.
                               if (length(ml) == 1 && is.na(ml)) {
                                 return(NA)
                               }
                               # If no error occurred, proceed as before.
                               names(ml) <- NULL
                               differences <- eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])
                               differences$nnode <- data$tes[[i]]$Nnode

                               # Save the differences to an RDS file with a timestamp-based filename
                               timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
                               timestamp <- paste0(timestamp, sample.int(1000, 1))
                               filename <- paste0("differences_", timestamp, ".rds")
                               saveRDS(differences, file = filename)

                               return(differences)
                             })
  return(diffs)
}


#' @export compute_accuracy_bd_ml_free
compute_accuracy_bd_ml_free <- function(dist_info, data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  mean_lambda <- eveGNN::compute_expected_mean(dist_info)
  mean_mu <- mean_lambda * 0.9
  diffs <- furrr::future_map(.x = seq_along(data$brts),
                             .f = function(i) {
                               ml <- DDD::bd_ML(
                                 brts = data$brts[[i]],
                                 initparsopt = data$pars[[i]],
                                 idparsopt = c(1, 2),
                                 tdmodel = 0,
                                 btorph = 0,
                                 soc = 2,
                                 cond = 1,
                                 num_cycles = Inf
                               )
                               # If an error occurred, ml will be NA and we return NA right away.
                               if (length(ml) == 1 && is.na(ml)) {
                                 return(NA)
                               }
                               # If no error occurred, proceed as before.
                               names(ml) <- NULL
                               differences <- eveGNN::all_differences(as.numeric(ml[1:2]), as.numeric(data$pars[[i]]))
                               differences$nnode <- data$tes[[i]]$Nnode

                               # Save the differences to an RDS file with a timestamp-based filename
                               timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
                               timestamp <- paste0(timestamp, sample.int(1000, 1))
                               filename <- paste0("differences_", timestamp, ".rds")
                               saveRDS(differences, file = filename)

                               return(differences)
                             })
  return(diffs)
}