args <- commandArgs(TRUE)
name <- args[1]

target_combo_pd <- eve::edd_combo_maker(
  la = 0.6,
  mu = 0,
  beta_n = 0,
  beta_phi = 0,
  age = 10,
  model = "dsce2",
  metric = "pd",
  offset = "simtime"
)

target_combo_ed <- eve::edd_combo_maker(
  la = 0.6,
  mu = 0,
  beta_n = 0,
  beta_phi = 0,
  age = 10,
  model = "dsce2",
  metric = "ed",
  offset = "none"
)

target_combo_nnd <- eve::edd_combo_maker(
  la = 0.6,
  mu = 0,
  beta_n = 0,
  beta_phi = 0,
  age = 10,
  model = "dsce2",
  metric = "nnd",
  offset = "none"
)

target_result_pd <- eve::edd_sim_rep(
  combo = target_combo_pd,
  history = FALSE,
  verbose = FALSE,
  nrep = 1000
)
target_result_ed <- eve::edd_sim_rep(
  combo = target_combo_ed,
  history = FALSE,
  verbose = FALSE,
  nrep = 1000
)
target_result_nnd <- eve::edd_sim_rep(
  combo = target_combo_nnd,
  history = FALSE,
  verbose = FALSE,
  nrep = 1000
)


export_to_gnn(target_result_pd, paste0(name, "_pd"))
export_to_gnn(target_result_ed, paste0(name, "_ed"))
export_to_gnn(target_result_nnd, paste0(name, "_nnd"))
