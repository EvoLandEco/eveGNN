args <- commandArgs(TRUE)

name <- args[1]
set <- as.numeric(args[2])
nrep <- as.numeric(args[3])

combo1 <- eve::edd_combo_maker(
  la = c(0.6),
  mu = c(0.1),
  beta_n = c(-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0),
  beta_phi = c(-0.04),
  age = c(10),
  model = "dsce2",
  metric = c("pd"),
  offset = c("simtime")
)

combo2 <- eve::edd_combo_maker(
  la = c(0.6),
  mu = c(0.1),
  beta_n = c(-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0),
  beta_phi = c(-0.04),
  age = c(10),
  model = "dsce2",
  metric = c("ed"),
  offset = c("none")
)

combo3 <- eve::edd_combo_maker(
  la = c(0.6),
  mu = c(0.1),
  beta_n = c(-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0),
  beta_phi = c(-0.04),
  age = c(10),
  model = "dsce2",
  metric = c("nnd"),
  offset = c("none")
)

combo <- c(combo1, combo2, combo3)

names(combo) <- as.character(seq(1, length(combo)))

if (!file.exists("params")) {
  params <- eve:::combo_to_tibble(combo)
  write.table(params, file.path(name, "params.txt"))
}

out <-
  eve::edd_sim_rep(
    combo = combo[[set]],
    history = TRUE,
    verbose = FALSE,
    nrep = nrep
  )

print(paste0("Starting parameter set ", set))

eve:::check_folder(name)
eve:::save_result(out, name, set)
