args <- commandArgs(TRUE)

i <- as.numeric(args[1])
name <- as.character(args[2])

data <- readRDS(file.path(name, "DDD_MLE_TES/MLE_DATA/ddd_mle.rds"))

setwd(name)
setwd("DDD_MLE_TES")

# Maximize the ML, let the MLE run in the best way possible
tryCatch(
  R.utils::withTimeout({
    misspec <- data$tas[[i]]$Nnode - data$tes[[i]]$Nnode
    ml <- DDD::dd_ML(
      brts = data$brts[[i]],
      initparsopt = data$pars[[i]],
      idparsopt = c(1, 2, 3),
      btorph = 0,
      soc = 2,
      cond = 1,
      ddmodel = 1,
      missnumspec = misspec,
      num_cycles = Inf
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
  }, timeout = 60000),  # in seconds
  TimeoutException = function(ex) {
    return(NA)  # Return NA or some other indication of timeout
  }
)

if (!dir.exists("NO_INIT")) {
  dir.create("NO_INIT")
}

setwd("NO_INIT")

# Start the MLE with no true initial parameters, simulate a random or worst-case scenario
# TODO: Should read from the config file, get the actual distribution of the parameters
tryCatch(
  R.utils::withTimeout({
    ml <- DDD::dd_ML(
      brts = data$brts[[i]],
      initparsopt = c(runif(1, 0, 3), runif(1, 0, 0.9), runif(1, 10, 1000)),
      idparsopt = c(1, 2, 3),
      btorph = 0,
      soc = 2,
      cond = 1,
      ddmodel = 1
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
  }, timeout = 22800),  # in seconds
  TimeoutException = function(ex) {
    return(NA)  # Return NA or some other indication of timeout
  }
)
