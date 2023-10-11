dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 1.0),
  list(distribution = "uniform", n = 1, min = 0, max = 0.4)
)

batch_100_10 <- batch_sim_ddd(dists, 100, 10, 1, 100)
batch_80_10 <- batch_sim_ddd(dists, 80, 10, 1, 100)
batch_60_10 <- batch_sim_ddd(dists, 60, 10, 1, 100)
batch_40_10 <- batch_sim_ddd(dists, 40, 10, 1, 100)
batch_20_10 <- batch_sim_ddd(dists, 20, 10, 1, 100)

diffs_100_10 <- compute_accuracy_dd_ml(dists, batch_100_10, "multisession", 6)
diffs_80_10 <- compute_accuracy_dd_ml(dists, batch_80_10)
diffs_60_10 <- compute_accuracy_dd_ml(dists, batch_60_10)
diffs_40_10 <- compute_accuracy_dd_ml(dists, batch_40_10)
diffs_20_10 <- compute_accuracy_dd_ml(dists, batch_20_10)
