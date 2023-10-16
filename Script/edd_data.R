args <- commandArgs(TRUE)

name <- args[1]
beta_n <- args[2]
batch <- args[3]
index <- args[4]

if (!dir.exists(name)) {
  dir.create(name)
}

# Get the current precise time
current_time <- Sys.time()

# Convert the current time to a numeric value
time_numeric <- as.numeric(current_time)

# Generate a random value
random_value <- sample(1:1000000, 1)

# Combine the time-based numeric value and the random value to create a seed
seed_value <- time_numeric + random_value + batch

# Set the seed
set.seed(seed_value)

setwd(name)
dir.create("EVE_TES")
setwd("EVE_TES")

eve_tes_list <- replicate(5, eveGNN::edd_sim_fix_n(n = 200, pars = c(0.6, 0.1, beta_n, -0.00025),
                                                     age = 10,
                                                     model = "dsce2",
                                                     metric = "pd",
                                                     offset = "simtime"), simplify = FALSE)
eve_tes_list <- purrr::transpose(eve_tes_list)

eveGNN::export_to_gnn_batch(data = eve_tes_list, name = index, batch = batch, batch_size = 5, which = "tes")
EVE_params <- data.frame(beta_n = beta_n, lambda = 0.6, mu = 0.1, age = 10, n = 200)
write.table(EVE_params, paste0("EVE_TES_params.txt"))


