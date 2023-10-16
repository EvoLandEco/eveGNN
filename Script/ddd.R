dists <- list(
  list(distribution = "uniform", n = 1, min = 0.5, max = 1.0),
  list(distribution = "uniform", n = 1, min = 0, max = 0.4)
)

batch_100_10 <- batch_sim_ddd(dists, 100, 10, 1, 100)
batch_80_10 <- batch_sim_ddd(dists, 80, 10, 1, 100)
batch_60_10 <- batch_sim_ddd(dists, 60, 10, 1, 100)
batch_40_10 <- batch_sim_ddd(dists, 40, 10, 1, 100)
batch_20_10 <- batch_sim_ddd(dists, 20, 10, 1, 100)

batch_100_list <- list()
for (i in 1:20) {
  batch_100_list[[i]] <- batch_sim_ddd(dists, 100, i, 1, 100)
}

batch_caps_list <- list()
j <- 1
for (i in seq(from = 5, to = 100, by = 5)) {
  batch_caps_list[[j]] <- batch_sim_ddd(dists, i, 20, 1, 100)
  j <- j + 1
}

diffs_100_list <- list()
{
  #diffs_100_10 <- compute_accuracy_dd_ml(dists, batch_100_10, "multisession", 6)
  #diffs_80_10 <- compute_accuracy_dd_ml(dists, batch_80_10, "multisession", 6)
  diffs_60_10 <- compute_accuracy_dd_ml(dists, batch_60_10, "multisession", 6)
  diffs_40_10 <- compute_accuracy_dd_ml(dists, batch_40_10, "multisession", 6)
  diffs_20_10 <- compute_accuracy_dd_ml(dists, batch_20_10, "multisession", 6)

  for (i in 1:20) {
    diffs_100_list[[i]] <- compute_accuracy_dd_ml(dists, batch_100_list[[i]], "multisession", 6)
  }
}

diffs_cap_list <- list()
error_indices <- c()
{
  # Loop over the sequence 1 to 20
  for (i in 1:20) {
    print(paste0("Starting index ", i))
    # Use tryCatch to handle any errors that occur when calling compute_accuracy_dd_ml
    result <- tryCatch(
    {
      # Attempt to compute the accuracy
      compute_accuracy_dd_ml(dists, batch_caps_list[[i]], "multisession", 8)
    },
      error = function(e) {
        # On error, record the index i that caused the error
        error_indices <- c(error_indices, i)
        # Return NULL so that result will be NULL, which will be stored in diffs_cap_list[[i]]
        NULL
      }
    )
    # Store the result (or NULL if an error occurred) in diffs_cap_list
    diffs_cap_list[[i]] <- result
  }
  # Save the workspace image
  save.image()

  if(length(error_indices) > 0) {
    cat("Errors occurred at indices:", error_indices, "\n")
  } else {
    cat("No errors occurred.\n")
  }
}

collapse_and_merge <- function(list_of_lists, step = 1) {
  # Initialize an empty data frame to store the results
  merged_df <- data.frame(Value = numeric(0), Group = integer(0))

  # Loop through each sublist, create a temporary data frame, and bind it to merged_df
  for(i in seq_along(list_of_lists)) {
    temp_df <- data.frame(Value = unlist(list_of_lists[[i]], use.names = FALSE), Group = i * step)
    merged_df <- rbind(merged_df, temp_df)
  }

  # Convert the Group column to a factor
  merged_df$Group <- factor(merged_df$Group)

  # Return the resulting data frame
  return(merged_df)
}

diffs_100_df <- collapse_and_merge(diffs_100_list)
diffs_caps_df <- collapse_and_merge(diffs_cap_list, step = 5)

alpha <- 0.95
diffs_100_df_stats <- diffs_100_df[-1968,] %>%
  dplyr::filter(!is.na(Value)) %>%
  dplyr::group_by(Group) %>%
  dplyr::summarise(
    Mean = mean(Value),
    QLow = quantile(Value, probs = 0.10, na.rm = TRUE),
    QHigh = quantile(Value, probs = 0.90, na.rm = TRUE),
    MeanDeviation = mean((Value - 0), na.rm = TRUE)
  )
diffs_100_df_stats$Group <- as.numeric(as.character(diffs_100_df_stats$Group))
diffs_100_df$Group <- as.numeric(as.character(diffs_100_df$Group))

compute_stats(diffs_caps_df[-diffs_caps_df_extremes,])
diffs_caps_df_stats <- diffs_caps_df[-diffs_caps_df_extremes,] %>%
  dplyr::filter(!is.na(Value)) %>%
  dplyr::group_by(Group) %>%
  dplyr::summarise(
    Mean = mean(Value),
    QLow = quantile(Value, probs = 0.10, na.rm = TRUE),
    QHigh = quantile(Value, probs = 0.90, na.rm = TRUE),
    MeanDeviation = mean((Value - 0), na.rm = TRUE)
  )
diffs_caps_df_stats$Group <- as.numeric(as.character(diffs_caps_df_stats$Group))
diffs_caps_df$Group <- as.numeric(as.character(diffs_caps_df$Group))

p_cap <- ggplot(diffs_100_df[-1968,]) +
  geom_hline(yintercept = 0.0,linewidth=2,linetype=2) +
  geom_smooth(data = diffs_100_df_stats, aes(x = Group, y = QHigh), linewidth=0.1, linetype=2, se = FALSE) +
  geom_smooth(data = diffs_100_df_stats, aes(x = Group, y = QLow), linewidth=0.1, linetype=2 , se = FALSE) +
  #geom_ribbon(data = diffs_100_df_stats,
              #aes(x = Group,
                  #ymin = fitted(loess(QLow ~ Group, data = diffs_100_df_stats)),
                  #ymax = fitted(loess(QHigh ~ Group, data = diffs_100_df_stats))),
              #alpha = 0.2) +
  geom_boxplot(aes(x = Group, y = Value, fill = Group, group = Group), outlier.shape = 1, outlier.alpha = 0.3) +
  viridis::scale_fill_viridis(option = "D", discrete = FALSE, limits = c(0, 20)) +
  coord_cartesian(ylim = c(-0.45, 0.3)) +
  labs(x = "Crown age", y = "Mean differences") +
  theme(legend.position = "none",
        aspect.ratio = 3/4,
        panel.background = element_rect(fill = "transparent", color = NA),
        legend.background = element_rect(fill = "transparent", color = NA))

diffs_caps_df_extremes <- c(which(diffs_caps_df$Value > 0.6), which(diffs_caps_df$Value < -0.6))

p_age <- ggplot(diffs_caps_df[-diffs_caps_df_extremes,]) +
  geom_hline(yintercept = 0.0,linewidth=2,linetype=2) +
  geom_smooth(data = diffs_caps_df_stats, aes(x = Group, y = QHigh), linewidth=0.1, linetype=2,se = FALSE) +
  geom_smooth(data = diffs_caps_df_stats, aes(x = Group, y = QLow), linewidth=0.1, linetype=2 , se = FALSE) +
  #geom_ribbon(data = diffs_caps_df_stats,
   #           aes(x = Group,
    #              ymin = fitted(loess(QLow ~ Group, data = diffs_caps_df_stats)),
     #             ymax = fitted(loess(QHigh ~ Group, data = diffs_caps_df_stats))),
      #        alpha = 0.2) +
  geom_boxplot(aes(x = Group, y = Value, fill = Group, group = Group), outlier.shape = 1, outlier.alpha = 0.3) +
  viridis::scale_fill_viridis(option = "D", discrete = FALSE, limits = c(0, 100)) +
  #coord_cartesian(ylim = c(-0.45, 0.3)) +
  labs(x = "Carrying capacity", y = "Mean differences") +
  theme(legend.position = "none",
        aspect.ratio = 3/4,
        panel.background = element_rect(fill = "transparent", color = NA),
        legend.background = element_rect(fill = "transparent", color = NA))

dataset %>% tidyr::gather(key = "Metric", value = "Value", -Epoch) %>%
  ggplot() + geom_line(aes(Epoch, Value, color = Metric)) +
  facet_wrap(~ Metric, scales = "free_y")
ggplot(dataset) + geom_area(aes(Epoch, fitted(loess(Train_Accuracy ~ Epoch))))

p_age + p_cap + patchwork::plot_layout(ncol = 2, guides = "collect") +
  patchwork::plot_annotation(title = "MLE Accuracy",
    subtitle = "Mean differences between true and estimated parameters",
  caption = "Distributions of parameters: lambda = uniform(0.5, 1.0), mu = uniform(0.0, 0.4)\n
  Crown age was fixed to 20 while changing carrying capacity\n
  Carrying capacity was fixed to 100 while changing crown age.")