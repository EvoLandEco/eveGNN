args <- commandArgs(TRUE)

i <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "DDD_MLE_TES/MLE_DATA/ddd_mle.rds"))

tryCatch(
  R.utils::withTimeout({
    ml <- DDD::dd_ML(
      brts = data$brts[[i]],
      initparsopt = data$pars[[i]],
      idparsopt = c(1, 2, 3),
      btorph = 0,
      soc = 2,
      cond = 1,
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
    differences$nnode <- data$tes[[i]]$Nnode

    # Save the differences to an RDS file with a timestamp-based filename
    timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
    timestamp <- paste0(timestamp, sample.int(1000, 1))
    filename <- paste0("differences_", timestamp, ".rds")
    saveRDS(differences, file = filename)

    return(differences)
  }, timeout = 3500),  # in seconds
  TimeoutException = function(ex) {
    return(NA)  # Return NA or some other indication of timeout
  }
)
