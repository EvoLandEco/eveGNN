#' @export compute_accuracy_pbd_ml_free
compute_accuracy_pbd_ml_free <- function(data, strategy = "sequential", workers = 1) {
  eve:::check_parallel_arguments(strategy, workers)
  diffs <- furrr::future_map(.x = seq_along(data$brts),
                             .f = function(i) {
                               ml <- PBD::pbd_ML(
                                 brts = data$brts[[i]],
                                 idparsopt = c(1, 2, 3),
                                 exteq = 1,
                                 btorph = 0,
                                 soc = 2,
                                 # cond = 0,
                                 verbose = FALSE
                               )
                               # If an error occurred, ml will be NA and we return NA right away.
                               if (length(ml) == 1 && is.na(ml)) {
                                 return(NA)
                               }
                               # If no error occurred, proceed as before.
                               reordered_ml <- c(as.numeric(ml$b),
                                                 as.numeric(ml$lambda_1),
                                                 as.numeric(ml$b),
                                                 as.numeric(ml$mu_1),
                                                 as.numeric(ml$mu_2))
                               differences <- eveGNN::all_differences(reordered_ml, data$pars[[i]])

                               # Save the differences to an RDS file with a timestamp-based filename
                               timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
                               filename <- paste0("differences_", timestamp, ".rds")
                               saveRDS(differences, file = filename)

                               return(differences)
                             })
  return(diffs)
}
