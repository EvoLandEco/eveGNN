#' @export compute_accuracy_pbd_ml_free
compute_accuracy_pbd_ml_free <- function(dist_info, data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  mean_pars <- eveGNN::compute_expected_mean(dist_info)
  diffs <- furrr::future_map(.x = seq_along(data$brts),
                             .f = function(i) {
                               ml <- pbd_ML(
                                 brts = data$brts[[i]],
                                 initparsopt = mean_pars,
                                 idparsopt = c(1, 2, 3),
                                 btorph = 0,
                                 soc = 2,
                                 cond = 0,
                                 ddmodel = 1,
                                 num_cycles = 1
                               )
                               # If an error occurred, ml will be NA and we return NA right away.
                               if (length(ml) == 1 && is.na(ml)) {
                                 return(NA)
                               }
                               # If no error occurred, proceed as before.
                               names(ml) <- NULL
                               differences <- eveGNN::all_differences(as.numeric(ml[1:3]), data$pars[[i]])

                               # Save the differences to an RDS file with a timestamp-based filename
                               timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
                               filename <- paste0("differences_", timestamp, ".rds")
                               saveRDS(differences, file = filename)

                               return(differences)
                             })
  return(diffs)
}
