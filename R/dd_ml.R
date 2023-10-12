#' @export compute_accuracy_dd_ml
compute_accuracy_dd_ml <- function(dist_info, data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  mean_pars <- compute_expected_mean(dist_info)
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
                               mean_difference(as.numeric(ml[1:3]), data$pars[[i]])
                             })
  return(diffs)
}
