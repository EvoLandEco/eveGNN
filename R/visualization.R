#' @export plot_model_performance
plot_model_performance <- function(data, group_var = stop("Grouping variable not provided."),
                                   class = stop("Classes not provided."),
                                   legend_title = NULL,
                                   accuracy_ref = NULL, loss_ref = NULL) {
  loss_plot <- ggplot2::ggplot(data) +
    ggplot2::geom_line(ggplot2::aes_string(x = "Epoch", y = "Loss", color = group_var)) +
    ggplot2::geom_hline(yintercept = loss_ref, linetype = "dashed", color = "black") +
    #ggplot2::ylim(0, loss_ref) +
    ggplot2::annotate("text",
                      x = median(data$Epoch),
                      y = loss_ref * 1.05,
                      label = paste0("Loss = ", loss_ref),
                      color = "black", size = 3) +
    # Add labels at the end of the line
    # ggplot2::geom_text(data = dplyr::filter(data, Epoch == max(Epoch)),
    #                    ggplot2::aes(x = Epoch, y = Loss, label = cap),
    #                    hjust = 0, nudge_x = 0.1) +
    ggplot2::guides(color = ggplot2::guide_legend(nrow = 2)) +
    ggplot2::labs(y = "Loss", color = legend_title) +
    ggplot2::theme(aspect.ratio = 5 / 6)

  accu_plot <- ggplot2::ggplot(data) +
    ggplot2::geom_line(ggplot2::aes_string(x = "Epoch", y = "Test_Accuracy_Overall", color = group_var)) +
    ggplot2::geom_hline(yintercept = accuracy_ref, linetype = "dashed", color = "black") +
    #ggplot2::ylim(accuracy_ref, 1) +
    ggplot2::annotate("text",
                      x = median(data$Epoch),
                      y = accuracy_ref * 1.05,
                      label = paste0("Accuracy = ", accuracy_ref),
                      color = "black",
                      size = 3) +
    # Add labels at the end of the line
    # ggplot2::geom_text(data = dplyr::filter(data, Epoch == max(Epoch)),
    #                    ggplot2::aes(x = Epoch, y = Test_Accuracy_Overall, label = cap),
    #                    hjust = 0, nudge_x = 0.1) +
    ggplot2::guides(color = ggplot2::guide_legend(nrow = 2)) +
    ggplot2::labs(y = "Overall Accuracy", color = legend_title) +
    ggplot2::theme(aspect.ratio = 5 / 6)

  class_plot_list <- list()

  for (i in 1:length(class)) {
    class_col_name <- paste0("Test_", (i - 1), "_Accuracy")
    class_plot_list[[i]] <- ggplot2::ggplot(data) +
      ggplot2::geom_line(ggplot2::aes_string(x = "Epoch", y = class_col_name, color = group_var)) +
      ggplot2::geom_hline(yintercept = accuracy_ref, linetype = "dashed", color = "black") +
      #ggplot2::ylim(0, loss_ref) +
      ggplot2::annotate("text",
                        x = median(data$Epoch),
                        y = accuracy_ref * 1.05,
                        label = paste0("Loss = ", accuracy_ref),
                        color = "black", size = 3) +
      # Add labels at the end of the line
      # ggplot2::geom_text(data = dplyr::filter(data, Epoch == max(Epoch)),
      #                    ggplot2::aes(x = Epoch, y = Test_Class0_Accuracy, label = cap),
      #                    hjust = 0, nudge_x = 0.1) +
      ggplot2::guides(color = ggplot2::guide_legend(nrow = 2)) +
      ggplot2::labs(y = paste0(class[i], " Accuracy"), color = legend_title) +
      ggplot2::theme(aspect.ratio = 5 / 6)
  }

  plot_list <- append(list(loss_plot, accu_plot), class_plot_list)
  patch <- patchwork::wrap_plots(plot_list) +
    patchwork::plot_layout(ncol = 2, guides = "collect") &
    ggplot2::theme(legend.position = 'bottom',
                   panel.background = ggplot2::element_rect(fill = "transparent", color = NA),
                   legend.background = ggplot2::element_rect(fill = "transparent", color = NA))

  patch + patchwork::plot_annotation(
    title = 'GNN Training Results',
    subtitle = paste0('Graph neural network classification of eve simulation results')
  )

  return(patch)
}


#' @export plot_umap
plot_umap <- function(dataset, sets, grouping_var, params, target_model, target_which, target_type) {
  all_plot_data <- list()

  for (i in sets) {
    all_plot_data[[length(all_plot_data) + 1]] <-
      extract_umap_data_by_metadata(dataset = dataset,
                                    target_model = target_model, target_which = target_which,
                                    target_set = i, target_type = target_type,
                                    target_stat = "umap")
  }

  plot_list <- list()
  epochs <- 0
  j <- 0
  for (data in all_plot_data) {
    j <- j + 1
    epochs <- unique(data$epoch)
    k <- 0
    for (i in sort(unlist(epochs))) {
      k <- k + 1
      plot_data <- extract_umap_data_by_epoch(data, i)
      validity <- lapply(plot_data, function(x) {
        length(x) == 1
      })
      if (FALSE %in% unlist(validity)) {
        stop("Invalid data")
      }
      plot_list[[length(plot_list) + 1]] <- ggplot2::ggplot(plot_data$data[[1]]) +
        ggtrace::geom_point_trace(ggplot2::aes(x = V1, y = V2,
                                               #color = as.factor(label),
                                               fill = as.factor(label))) +
        ggplot2::scale_fill_discrete(labels = c("BD Tree", "DDD Tree")) +
        ggplot2::theme(aspect.ratio = 1,
                       plot.margin = ggplot2::unit(c(0, 0, 0, 0), "pt")) +
        ggplot2::labs(x = "", y = "", fill = "Class")
      if (k %% length(epochs) == 1) {
        grouping_var_value <- params[sets[j], grouping_var]
        plot_list[[length(plot_list)]] <- plot_list[[length(plot_list)]] +
          ggplot2::labs(y = paste0(grouping_var_to_label(grouping_var), ": ", grouping_var_value))
      }
      if ((length(sets) * length(epochs)) - length(plot_list) < length(epochs)) {
        plot_list[[length(plot_list)]] <- plot_list[[length(plot_list)]] +
          ggplot2::labs(x = paste0("Epoch: ", i))
      }
    }
  }

  patch <- patchwork::wrap_plots(plot_list) + patchwork::plot_layout(ncol = length(epochs), guides = "collect") &
    ggplot2::theme(legend.position = 'bottom',
                   plot.margin = ggplot2::unit(c(5, 5, 5, 5), "pt"),
                   panel.background = ggplot2::element_rect(fill = "transparent", color = NA),
                   plot.background = ggplot2::element_rect(fill = "transparent", color = NA))

  return(patch)
}


#' @export plot_scatter_difference_by_pars_rel
plot_scatter_difference_by_pars_rel <- function(path = NULL, task = "BD", model = "diffpool", max_depth = 5, which = "nnode", abline_color = "red", abline_range = c(-100, 100)) {
  xvar <- NULL
  if (which == "nnode") {
    xvar <- "num_nodes"
    xlabel <- "Number of Nodes"
  }
  if (which == "lambda") {
    xvar <- "lambda"
    xlabel <- "Speciation rate"
  }
  if (which == "mu") {
    xvar <- "mu"
    xlabel <- "Extinction rate"
  }
  if (which == "cap") {
    xvar <- "cap"
    xlabel <- "Carrying capacity"
  }
  if (which == "netspe") {
    xvar <- "(lambda - mu)"
    xlabel <- "Net speciation rate"
  }
  plot_title <- NULL
  if (task == "BD") {
    plot_title <- "Birth-Death Model"
  } else if (task == "DDD") {
    plot_title <- "Diversity-Dependent-Diversification Model"
  } else if (task == "PBD") {
    plot_title <- "Protracted Birth-Death Model"
  } else if (task == "EVE") {
    plot_title <- "Evolutionary-Relatedness-Dependent Model"
  }

  plot_data_tes <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TES"), model_type = model, max_depth = max_depth)
  plot_data_tas <- NULL
  plot_data <- NULL
  if (task != "PBD") {
    plot_data_tas <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TAS"), model_type = model, max_depth = max_depth)
    plot_data <- rbind(plot_data_tes, plot_data_tas)
  } else {
    plot_data <- plot_data_tes
  }
  plot_data_mle <- NULL
  if (task != "EVE") {
    if (task == "DDD") {
      plot_data_mle <- load_separated_mle_result(path = path, task_type = task, model_type = model)
      plot_data_mle <- plot_data_mle %>%
        dplyr::filter(lambda_r_diff < 2000, lambda_r_diff > -2000, mu_r_diff < 2000, mu_r_diff > -2000, cap_r_diff < 2000, cap_r_diff > -2000) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -num_nodes, -lambda, -mu, -cap)
    } else if (task == "BD") {
      plot_data_mle <- load_full_mle_result(path = path, task_type = paste0(task, "_MLE_TES"), model_type = model)
      plot_data_mle <- plot_data_mle %>%
        dplyr::filter(lambda_r_diff < 2000, lambda_r_diff > -2000, mu_r_diff < 2000, mu_r_diff > -2000) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -num_nodes, -lambda, -mu)
    } else if (task == "PBD") {

    }
  }
  plot_data$Depth <- as.factor(plot_data$Depth)
  if (task == "BD") {
    plot_data <- plot_data %>%
      dplyr::select(-lambda_diff, -mu_diff, -lambda_pred, -mu_pred) %>%
      dplyr::filter(lambda_r_diff < 250, mu_r_diff < 250, lambda_r_diff > -250, mu_r_diff > -250) %>%
      tidyr::gather("Parameter", "Value", -Task, -Model, -Depth, -lambda, -mu, -num_nodes)
  } else if (task == "DDD") {
    plot_data <- plot_data %>%
      dplyr::select(-lambda_diff, -mu_diff, -lambda_pred, -mu_pred, -cap_diff, -cap_pred) %>%
      dplyr::filter(lambda_r_diff < 250, mu_r_diff < 250, lambda_r_diff > -250, mu_r_diff > -250) %>%
      tidyr::gather("Parameter", "Value", -Task, -Model, -Depth, -lambda, -mu, -num_nodes, -cap)
  }

  plot_list <- list()

  index <- 1

  for (i in unique(plot_data$Task)) {
    p <- ggplot2::ggplot(plot_data) +
      ggplot2::facet_grid(Depth ~ Parameter,
                          labeller =
                            ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed),
                                              Depth = ggplot2::as_labeller(~paste0(.x, "~'Layer(s)'"), ggplot2::label_parsed))) +
      ggpointdensity::geom_pointdensity(data = plot_data %>% dplyr::filter(Task == i),
                                        ggplot2::aes_string(xvar, "Value"), method = "kde2d") +
      ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
      ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
      viridis::scale_color_viridis(discrete = F) +
      ggplot2::theme(legend.position = "none",
                     plot.background = ggplot2::element_blank(),
                     panel.background = ggplot2::element_blank()) +
      ggplot2::ylim(-200, 200)

    if (length(unique(plot_data$Task)) == 1) {
      p <- p + ggplot2::labs(x = xlabel, y = "Relative difference (%)")
    } else {
      if (index == 1) {
        p <- p + ggplot2::labs(x = NULL, y = "Relative difference (%)")
      } else {
        p <- p + ggplot2::labs(x = xlabel, y = NULL)
      }
    }

    if (grepl("TES", i)) {
      p <- p + ggplot2::ggtitle(paste0("GNN ", " Extant trees"))
    } else if (grepl("TAS", i)) {
      p <- p + ggplot2::ggtitle(paste0("GNN ", " Full trees"))
    }

    plot_list[[length(plot_list) + 1]] <- p
    index <- index + 1
  }
  if (task != "EVE") {
    plot_list[[length(plot_list) + 1]] <- ggplot2::ggplot() +
      ggplot2::facet_grid(. ~ Parameter,
                          labeller = ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed))) +
      ggpointdensity::geom_pointdensity(data = plot_data_mle,
                                        ggplot2::aes_string(xvar, "Value")) +
      ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
      ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
      viridis::scale_color_viridis(discrete = F) +
      ggplot2::theme(legend.position = "none",
                     plot.background = ggplot2::element_blank(),
                     panel.background = ggplot2::element_blank()) +
      ggplot2::ggtitle(paste0("MLE ", " Extant trees")) +
      ggplot2::labs(x = NULL, y = NULL)
  }

  return(patchwork::wrap_plots(plotlist = plot_list, ncol = length(plot_list)) +
           patchwork::plot_annotation(title = plot_title))
}


#' @export plot_scatter_difference_by_pars_rel_by_layer
plot_scatter_difference_by_pars_rel_by_layer <- function(path = NULL, task = "BD", model = "diffpool", depth = 5, abline_color = "red", abline_range = c(-100, 100)) {
  xlabel_list <- NULL
  if (task == "BD") {
    xvar_list <- c("lambda", "mu", "(lambda - mu)", "num_nodes")
    xlabel_list <- c("Speciation rate", "Extinction rate", "Net speciation rate", "Number of nodes")
  }
  if (task == "DDD") {
    xvar_list <- c("lambda", "mu", "cap", "num_nodes")
    xlabel_list <- c("Speciation rate", "Extinction rate", "Carrying capacity", "Number of nodes")
  }
  if (task == "PBD") {
    xvar_list <- c("lambda1", "lambda2", "lambda3", "mu1", "mu2", "num_nodes")
    xlabel_list <- c("Speciation rate of good species",
                     "Speciation completion rate",
                     "Speciation rate of incipient species",
                     "Extinction rate of good species",
                     "Extinction rate of incipient species",
                     "Number of nodes")
  }
  if (task == "EVE") {
    xvar_list <- c("lambda", "mu", "beta_n", "beta_phi", "num_nodes")
    xlabel_list <- c("Speciation rate", "Extinction rate",
                     "Species richness effect", "Evolutionary relatedness effect",
                     "Number of nodes")
  }

  plot_title <- NULL
  if (task == "BD") {
    plot_title <- "Relative Difference (Birth-Death Model)"
  } else if (task == "DDD") {
    plot_title <- "Relative Difference (Diversity-Dependent-Diversification Model)"
  } else if (task == "PBD") {
    plot_title <- "Relative Difference (Protracted Birth-Death Model)"
  } else if (task == "EVE") {
    plot_title <- "Relative Difference (Evolutionary-Relatedness-Dependent Model)"
  }

  plot_data_tes <- load_final_difference_by_layer(path = path, task_type = paste0(task, "_FREE_TES"), model_type = model, depth = depth)
  plot_data_tas <- NULL
  plot_data <- NULL
  if (task != "PBD") {
    plot_data_tas <- load_final_difference_by_layer(path = path, task_type = paste0(task, "_FREE_TAS"), model_type = model, depth = depth)
    plot_data <- rbind(plot_data_tes, plot_data_tas)
  } else {
    plot_data <- plot_data_tes
  }

  plot_data_mle <- NULL
  if (task != "EVE") {
    if (task == "DDD") {
      plot_data_mle <- load_separated_mle_result(path = path, task_type = task, model_type = model)
      plot_data_mle <- plot_data_mle %>%
        dplyr::select(-lambda_a_diff, -mu_a_diff, -cap_a_diff) %>%
        dplyr::filter(lambda_r_diff < 2000, lambda_r_diff > -2000, mu_r_diff < 2000, mu_r_diff > -2000, cap_r_diff < 2000, cap_r_diff > -2000) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -num_nodes, -lambda, -mu, -cap)
    } else if (task == "BD") {
      plot_data_mle <- load_full_mle_result(path = path, task_type = paste0(task, "_MLE_TES"), model_type = model)
      plot_data_mle <- plot_data_mle %>%
        dplyr::filter(lambda_r_diff < 2000, lambda_r_diff > -2000, mu_r_diff < 2000, mu_r_diff > -2000) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -num_nodes, -lambda, -mu)
    } else if (task == "PBD") {
      plot_data_mle <- load_separated_mle_result(path = path, task_type = task, model_type = model)
      plot_data_mle <- plot_data_mle %>%
        dplyr::select(-lambda1_a_diff, -lambda2_a_diff, -lambda3_a_diff, -mu1_a_diff, -mu2_a_diff) %>%
        dplyr::filter(lambda1_r_diff < 2000, lambda1_r_diff > -2000,
                      lambda2_r_diff < 2000, lambda2_r_diff > -2000,
                      lambda3_r_diff < 2000, lambda3_r_diff > -2000,
                      mu1_r_diff < 2000, mu1_r_diff > -2000,
                      mu2_r_diff < 2000, mu2_r_diff > -2000) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -num_nodes, -Depth, -lambda1, -lambda2, -lambda3, -mu1, -mu2)
    }
  }

  plot_data$Depth <- as.factor(plot_data$Depth)
  if (task == "BD") {
    plot_data <- plot_data %>%
      dplyr::select(-lambda_diff, -mu_diff, -lambda_pred, -mu_pred, -lambda_a_diff, -mu_a_diff) %>%
      dplyr::filter(lambda_r_diff < 350, mu_r_diff < 350, lambda_r_diff > -350, mu_r_diff > -350) %>%
      tidyr::gather("Parameter", "Value", -Task, -Model, -Depth, -lambda, -mu, -num_nodes)
  } else if (task == "DDD") {
    plot_data <- plot_data %>%
      dplyr::select(-lambda_diff, -mu_diff, -lambda_pred, -mu_pred, -cap_diff, -cap_pred, -lambda_a_diff, -mu_a_diff, -cap_a_diff) %>%
      dplyr::filter(lambda_r_diff < 350, mu_r_diff < 350, lambda_r_diff > -350, mu_r_diff > -350) %>%
      tidyr::gather("Parameter", "Value", -Task, -Model, -Depth, -lambda, -mu, -num_nodes, -cap)
  } else if (task == "PBD") {
    plot_data <- plot_data %>%
      dplyr::select(-lambda1_diff, -lambda2_diff, -lambda3_diff, -mu1_diff, -mu2_diff, -lambda1_pred, -lambda2_pred, -lambda3_pred, -mu1_pred, -mu2_pred, -lambda1_a_diff, -lambda2_a_diff, -lambda3_a_diff, -mu1_a_diff, -mu2_a_diff) %>%
      dplyr::filter(lambda1_r_diff < 350, lambda2_r_diff < 350, lambda3_r_diff < 350, mu1_r_diff < 350, mu2_r_diff < 350,
                    lambda1_r_diff > -350, lambda2_r_diff > -350, lambda3_r_diff > -350, mu1_r_diff > -350, mu2_r_diff > -350) %>%
      tidyr::gather("Parameter", "Value", -Task, -Model, -Depth, -lambda1, -lambda2, -lambda3, -mu1, -mu2, -num_nodes)
  } else if (task == "EVE") {
    plot_data <- plot_data %>%
      dplyr::select(-lambda_diff, -mu_diff, -beta_n_diff, -beta_phi_diff, -lambda_pred, -mu_pred, -beta_n_pred, -beta_phi_pred, -lambda_a_diff, -mu_a_diff, -beta_n_a_diff, -beta_phi_a_diff) %>%
      dplyr::filter(lambda_r_diff < 350, mu_r_diff < 350, beta_n_r_diff < 350, beta_phi_r_diff < 350,
                    lambda_r_diff > -350, mu_r_diff > -350, beta_n_r_diff > -350, beta_phi_r_diff > -350) %>%
      tidyr::gather("Parameter", "Value", -Task, -Model, -Depth, -lambda, -mu, -beta_n, -beta_phi, -num_nodes)
  }

  plot_list <- list()
  flag <- FALSE
  for (i in unique(plot_data$Task)) {
    plot_sub_list <- list()
    index <- 1

    for (j in seq_len(length(xvar_list))) {
      p <- ggplot2::ggplot(plot_data) +
        ggplot2::facet_wrap(. ~ Parameter,
                            labeller =
                              ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed)),
                            nrow = 1) +
        ggpointdensity::geom_pointdensity(data = plot_data %>% dplyr::filter(Task == i),
                                          ggplot2::aes_string(xvar_list[j], "Value"), method = "kde2d") +
        ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
        ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
        #viridis::scale_color_viridis(discrete = F) +
        nord::scale_color_nord(palette = "afternoon_prarie", discrete = F) +
        ggplot2::scale_y_continuous(labels = function(x) paste0(x, "%")) +
        ggplot2::labs(y = NULL, x = NULL) +
        ggplot2::theme(legend.position = "none",
                       plot.background = ggplot2::element_blank(),
                       panel.background = ggplot2::element_blank()) +
        ggplot2::coord_cartesian(ylim = c(-220, 220))

      if (index == 1) {
        if (grepl("TES", i)) {
          p <- p + ggplot2::ggtitle(paste0("GNN ", " Extant trees"))
        } else if (grepl("TAS", i)) {
          p <- p + ggplot2::ggtitle(paste0("GNN ", " Full trees"))
        }
      }

      if (flag == FALSE) {
        p <- p + ggplot2::labs(y = grouping_var_to_label(xvar_list[j]))
      }

      plot_sub_list[[length(plot_sub_list) + 1]] <- p
      index <- index + 1
    }

    plot_list[[length(plot_list) + 1]] <- patchwork::wrap_plots(plotlist = plot_sub_list,
                                                                ncol = 1,
                                                                byrow = FALSE,
                                                                guides = "collect")

    flag <- TRUE
  }

  plot_sub_list <- list()
  flag <- FALSE

  for (j in seq_len(length(xvar_list))) {
    if (task != "EVE") {
      p <- ggplot2::ggplot() +
        ggplot2::facet_wrap(. ~ Parameter,
                            labeller = ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed)),
                            nrow = 1) +
        ggpointdensity::geom_pointdensity(data = plot_data_mle,
                                          ggplot2::aes_string(xvar_list[j], "Value")) +
        ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
        ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
        #viridis::scale_color_viridis(discrete = F) +
        nord::scale_color_nord(palette = "afternoon_prarie", discrete = F) +
        ggplot2::scale_y_continuous(labels = function(x) paste0(x, "%")) +
        ggplot2::theme(legend.position = "none",
                       plot.background = ggplot2::element_blank(),
                       panel.background = ggplot2::element_blank()) +
        ggplot2::labs(x = NULL, y = NULL)

      if (flag == FALSE) {
        p <- p + ggplot2::ggtitle(paste0("MLE ", " Extant trees"))
      }

      plot_sub_list[[length(plot_sub_list) + 1]] <- p
    }

    flag <- TRUE
  }

  plot_list[[length(plot_list) + 1]] <- patchwork::wrap_plots(plotlist = plot_sub_list,
                                                              ncol = 1,
                                                              byrow = FALSE,
                                                              guides = "collect")

  return(patchwork::wrap_plots(plotlist = plot_list, ncol = length(plot_list)) +
           patchwork::plot_annotation(title = plot_title))
}


#' @export plot_boxplot_difference
plot_boxplot_difference <- function(path = NULL, model = "diffpool", max_depth = 5) {
  task_list <- c("BD", "DDD", "PBD")
  n_task <- 1
  final_plots <- list()

  for(task in task_list) {
    plot_data_tes <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TES"), model_type = model, max_depth = max_depth)
    plot_data_tes_val <- load_final_difference(path = path, task_type = paste0(task, "_VAL_TES"), model_type = model, max_depth = max_depth)
    names(plot_data_tes_val)[names(plot_data_tes_val) == "nodes"] <- "num_nodes"

    plot_data_tas <- NULL
    plot_data_tas_val <- NULL
    plot_data <- rbind(plot_data_tes, plot_data_tes_val)

    if (task != "PBD") {
      plot_data_tas <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TAS"), model_type = model, max_depth = max_depth)
      plot_data_tas_val <- load_final_difference(path = path, task_type = paste0(task, "_VAL_TAS"), model_type = model, max_depth = max_depth)
      names(plot_data_tas_val)[names(plot_data_tas_val) == "nodes"] <- "num_nodes"
      plot_data <- rbind(plot_data, plot_data_tas, plot_data_tas_val)
    } else {

    }

    plot_data_mle <- NULL

    if (task != "EVE") {
      if (task == "DDD") {
        plot_data_mle <- load_separated_mle_result(path = path, task_type = task, model_type = model)
        plot_data_mle <- plot_data_mle %>%
          dplyr::filter(lambda_r_diff < 2000, lambda_r_diff > -2000, mu_r_diff < 2000, mu_r_diff > -2000, cap_r_diff < 2000, cap_r_diff > -2000) %>%
          dplyr::select(lambda_r_diff, mu_r_diff, cap_r_diff, Task, Model) %>%
          dplyr::mutate(Depth = NA) %>%
          tidyr::gather("Parameter", "Value", -Task, -Model, -Depth)
      } else if (task == "BD") {
        plot_data_mle <- load_full_mle_result(path = path, task_type = paste0(task, "_MLE_TES"), model_type = model)
        plot_data_mle <- plot_data_mle %>%
          dplyr::filter(lambda_r_diff < 1000, lambda_r_diff > -1000, mu_r_diff < 1500, mu_r_diff > -1500) %>%
          dplyr::select(lambda_r_diff, mu_r_diff, Task, Model) %>%
          dplyr::mutate(Depth = NA) %>%
          tidyr::gather("Parameter", "Value", -Task, -Model, -Depth)
      } else if (task == "PBD") {
        plot_data_mle <- load_separated_mle_result(path = path, task_type = task, model_type = model)
        plot_data_mle <- plot_data_mle %>%
          dplyr::filter(lambda1_r_diff < 2000, lambda1_r_diff > -2000,
                        lambda2_r_diff < 2000, lambda2_r_diff > -2000,
                        lambda3_r_diff < 2000, lambda3_r_diff > -2000,
                        mu1_r_diff < 2000, mu1_r_diff > -2000,
                        mu2_r_diff < 2000, mu2_r_diff > -2000) %>%
          dplyr::select(lambda1_r_diff, lambda2_r_diff, lambda3_r_diff, mu1_r_diff, mu2_r_diff, Task, Model) %>%
          dplyr::mutate(Depth = NA) %>%
          tidyr::gather("Parameter", "Value", -Task, -Model, -Depth)
      }
    }

    plot_title <- NULL

    if (task == "BD") {
      plot_title <- "BD"
    } else if (task == "DDD") {
      plot_title <- "DDD"
    } else if (task == "PBD") {
      plot_title <- "PBD"
    } else if (task == "EVE") {
      plot_title <- "EVE"
    }

    plot_data$Depth <- as.factor(plot_data$Depth)
    plot_data_mle$Depth <- as.factor(plot_data_mle$Depth)
    if (task == "BD") {
      plot_data <- plot_data %>%
        dplyr::select(lambda_r_diff, mu_r_diff, Depth, Model, Task) %>%
        dplyr::filter(lambda_r_diff < 2500, mu_r_diff < 2500, lambda_r_diff > -2500, mu_r_diff > -2500) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -Depth)
    } else if (task == "DDD") {
      plot_data <- plot_data %>%
        dplyr::select(lambda_r_diff, mu_r_diff, cap_r_diff, Depth, Model, Task) %>%
        dplyr::filter(lambda_r_diff < 2500, mu_r_diff < 2500, lambda_r_diff > -2500, mu_r_diff > -2500) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -Depth)
    } else if (task == "PBD") {
      plot_data <- plot_data %>%
        dplyr::filter(lambda1_r_diff < 2000, lambda1_r_diff > -2000,
                      lambda2_r_diff < 2000, lambda2_r_diff > -2000,
                      lambda3_r_diff < 2000, lambda3_r_diff > -2000,
                      mu1_r_diff < 2000, mu1_r_diff > -2000,
                      mu2_r_diff < 2000, mu2_r_diff > -2000) %>%
        dplyr::select(lambda1_r_diff, lambda2_r_diff, lambda3_r_diff, mu1_r_diff, mu2_r_diff, Depth, Task, Model) %>%
        tidyr::gather("Parameter", "Value", -Task, -Model, -Depth)
    }

    plot_data_all <- dplyr::bind_rows(plot_data, plot_data_mle) %>%
      dplyr::mutate(Flag = dplyr::case_when(
        grepl("FREE", Task) ~ "Free",
        grepl("VAL", Task) ~ "Val",
        grepl("MLE", Task) ~ "MLE"))

    plot <- list()
    index <- 1
    for (i in unique(plot_data_all$Parameter)) {
      p <- ggplot2::ggplot(data = plot_data_all %>%
        dplyr::filter(Parameter == i) %>%
        dplyr::mutate(Tag = difference_var_to_label(i))) +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
        ggplot2::geom_boxplot(ggplot2::aes(Task, Value, fill = Flag), outlier.shape = NA) +
        ggplot2::facet_wrap(~ Tag, labeller = ggplot2::label_parsed) +
        ggplot2::scale_y_continuous(labels = function(x) paste0(x, "%")) +
        ggplot2::labs(x = NULL, y = NULL, fill = "Type") +
        nord::scale_fill_nord(palette = "aurora", discrete = T, labels = c("Test", "MLE", "Validation")) +
        ggplot2::guides(fill = ggplot2::guide_legend(nrow = 1, byrow = TRUE)) +
        ggplot2::theme(plot.tag.position = "top",
                       panel.background = ggplot2::element_blank())

      if (task == "BD") {
        p <- p + ggplot2::scale_x_discrete(labels = c("T-TAS", "T-TES", "MLE-TES", "V-TAS", "V-TES"))
        if (i == "lambda_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-250, 250))
        } else if (i == "mu_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-500, 250))
        }
      } else if (task == "DDD") {
        p <- p + ggplot2::scale_x_discrete(labels = c("T-TAS", "T-TES", "MLE-TES", "V-TAS", "V-TES"))
        if (i == "lambda_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-300, 250))
        } else if (i == "mu_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-500, 250))
        } else if (i == "cap_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-250, 250))
        }
      } else if (task == "PBD") {
        p <- p + ggplot2::scale_x_discrete(labels = c("T-TES", "MLE-TES", "V-TES"))
        if (i == "lambda1_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-250, 250))
        } else if (i == "lambda2_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-2000, 150))
        } else if (i == "lambda3_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-300, 250))
        } else if (i == "mu1_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-1000, 250))
        } else if (i == "mu2_r_diff") {
          p <- p + ggplot2::coord_cartesian(ylim = c(-1200, 250))
        }
      }

      if (index == 1) {
        p <- p + ggplot2::labs(y = plot_title)
      }

      index <- index + 1

      plot[[length(plot) + 1]] <- p
    }

    if (n_task != length(task_list)) {
      final_plots[[length(final_plots) + 1]] <-
        patchwork::wrap_plots(plotlist = plot, nrow = 1, guides = "collect") &
          ggplot2::theme(legend.position = "none")
    } else {
      final_plots[[length(final_plots) + 1]] <-
        patchwork::wrap_plots(plotlist = plot, nrow = 1, guides = "collect") &
        ggplot2::theme(legend.position = "bottom")
    }

    n_task <- n_task + 1
  }

  out <- patchwork::wrap_plots(plotlist = final_plots, ncol = 1) +
    patchwork::plot_annotation(title = "Relative Difference by Models and Estimation Methods")

  return(out)
}


#' @export plot_scatter_difference_in_out_sample_rel
plot_scatter_difference_in_out_sample_rel <- function(path = NULL,
                                                  task = "BD",
                                                  model = "diffpool",
                                                  max_depth = 5,
                                                  which = "lambda",
                                                  abline_color = "red",
                                                  abline_range = c(-100, 100),
                                                  within_range = NULL) {
  xvar <- NULL
  if (which == "lambda") {
    xvar <- "lambda"
    xlabel <- "Speciation rate"
  }
  if (which == "mu") {
    xvar <- "mu"
    xlabel <- "Extinction rate"
  }
  if (which == "cap") {
    xvar <- "cap"
    xlabel <- "Carrying capacity"
  }
  if (which == "netspe") {
    xvar <- "(lambda - mu)"
    xlabel <- "Net speciation rate"
  }

  plot_title <- NULL
  if (task == "BD") {
    plot_title <- "Birth-Death Model"
  } else if (task == "DDD") {
    plot_title <- "Diversity-Dependent-Diversification Model"
  } else if (task == "PBD") {
    plot_title <- "Protracted Birth-Death Model"
  } else if (task == "EVE") {
    plot_title <- "Evolutionary-Relatedness-Dependent Model"
  }

  plot_data_tes_val <- load_final_difference(path = path, task_type = paste0(task, "_VAL_TES"), model_type = model, max_depth = max_depth)
  plot_data_tes <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TES"), model_type = model, max_depth = max_depth)
  plot_data_tes_val$nodes <- NULL
  plot_data_tes_val$Type <- "Validation"
  plot_data_tes_val$Task <- "TES"
  plot_data_tes$num_nodes <- NULL
  plot_data_tes$Type <- "Test"
  plot_data_tes$Task <- "TES"

  plot_data_tas_val <- NULL
  plot_data <- NULL
  if (task != "PBD") {
    plot_data_tas_val <- load_final_difference(path = path, task_type = paste0(task, "_VAL_TAS"), model_type = model, max_depth = max_depth)
    plot_data_tas <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TAS"), model_type = model, max_depth = max_depth)
    plot_data_tas_val$nodes <- NULL
    plot_data_tas_val$Type <- "Validation"
    plot_data_tas_val$Task <- "TAS"
    plot_data_tas$num_nodes <- NULL
    plot_data_tas$Type <- "Test"
    plot_data_tas$Task <- "TAS"
    plot_data <- rbind(plot_data_tes, plot_data_tas, plot_data_tes_val, plot_data_tas_val)
  } else {
    plot_data <- c(plot_data_tes, plot_data_tes_val)
  }

  if (task == "BD") {
    pars_list <- c("lambda", "mu", "Depth", "Model", "Task", "Type")
    data_var_list <- c("lambda", "mu", "lambda_r_diff", "mu_r_diff", "Depth", "Model", "Task", "Type")
  } else if (task == "DDD") {
    pars_list <- c("lambda", "mu", "cap", "Depth", "Model", "Task", "Type")
    data_var_list <- c("lambda", "mu", "cap", "lambda_r_diff", "mu_r_diff", "cap_r_diff", "Depth", "Model", "Task", "Type")
  } else if (task == "PBD") {

  }

  plot_data <- plot_data %>%
    dplyr::select(dplyr::all_of(data_var_list)) %>%
    tidyr::gather("Parameter", "Value", -dplyr::all_of(pars_list))

  plot_list <- list()

  index <- 1

  for (i in unique(plot_data$Task)) {
    p <- ggplot2::ggplot(plot_data) +
      ggplot2::facet_grid(Depth ~ Parameter,
                          labeller =
                            ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed),
                                              Depth = ggplot2::as_labeller(~paste0(.x, "~'Layer(s)'"), ggplot2::label_parsed))) +
      ggplot2::geom_point(data = plot_data %>% dplyr::filter(Task == i),
                          ggplot2::aes_string(xvar, "Value", color = "Type", alpha = "Type")) +
      ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
      ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
      ggplot2::scale_alpha_manual(name = "Type", values = c(0.5, 0.1)) +
      viridis::scale_color_viridis(discrete = T) +
      ggplot2::scale_y_continuous(labels = function(x) paste0(x, "%")) +
      ggplot2::theme(legend.position = "none",
                     plot.background = ggplot2::element_blank(),
                     panel.background = ggplot2::element_blank()) +
      ggplot2::coord_cartesian(ylim = c(-250, 250))

    if (length(unique(plot_data$Task)) == 1) {
      p <- p + ggplot2::labs(x = xlabel, y = "Relative difference (%)")
    } else {
      if (index == 1) {
        p <- p + ggplot2::labs(x = NULL, y = "Relative difference (%)")
      } else {
        p <- p + ggplot2::labs(x = xlabel, y = NULL)
      }
    }

    if (grepl("TES", i)) {
      p <- p + ggplot2::ggtitle(paste0("GNN ", " Extant trees"))
    } else if (grepl("TAS", i)) {
      p <- p + ggplot2::ggtitle(paste0("GNN ", " Full trees"))
    }

    p <- p +
      ggplot2::geom_vline(xintercept = within_range[1], linetype = "dashed", color = "blue") +
      ggplot2::geom_vline(xintercept = within_range[2], linetype = "dashed", color = "blue")

    plot_list[[length(plot_list) + 1]] <- p
    index <- index + 1
  }

  out <- patchwork::wrap_plots(plotlist = plot_list, ncol = length(plot_list)) +
    patchwork::plot_annotation(title = plot_title)

  return(out)
}


#' @export plot_scatter_difference_in_out_sample_abs
plot_scatter_difference_in_out_sample_abs <- function(path = NULL,
                                                      task = "BD",
                                                      model = "diffpool",
                                                      max_depth = 5,
                                                      which = "lambda",
                                                      abline_color = "red",
                                                      abline_range = c(-0.05, 0.05),
                                                      within_range = NULL) {
  xvar <- NULL
  if (which == "lambda") {
    xvar <- "lambda"
    xlabel <- "Speciation rate"
  }
  if (which == "mu") {
    xvar <- "mu"
    xlabel <- "Extinction rate"
  }
  if (which == "cap") {
    xvar <- "cap"
    xlabel <- "Carrying capacity"
  }
  if (which == "netspe") {
    xvar <- "(lambda - mu)"
    xlabel <- "Net speciation rate"
  }

  plot_title <- NULL
  if (task == "BD") {
    plot_title <- "Birth-Death Model"
  } else if (task == "DDD") {
    plot_title <- "Diversity-Dependent-Diversification Model"
  } else if (task == "PBD") {
    plot_title <- "Protracted Birth-Death Model"
  } else if (task == "EVE") {
    plot_title <- "Evolutionary-Relatedness-Dependent Model"
  }

  plot_data_tes_val <- load_final_difference(path = path, task_type = paste0(task, "_VAL_TES"), model_type = model, max_depth = max_depth)
  plot_data_tes <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TES"), model_type = model, max_depth = max_depth)
  plot_data_tes_val$nodes <- NULL
  plot_data_tes_val$Type <- "Validation"
  plot_data_tes_val$Task <- "TES"
  plot_data_tes$num_nodes <- NULL
  plot_data_tes$Type <- "Test"
  plot_data_tes$Task <- "TES"

  plot_data_tas_val <- NULL
  plot_data <- NULL
  if (task != "PBD") {
    plot_data_tas_val <- load_final_difference(path = path, task_type = paste0(task, "_VAL_TAS"), model_type = model, max_depth = max_depth)
    plot_data_tas <- load_final_difference(path = path, task_type = paste0(task, "_FREE_TAS"), model_type = model, max_depth = max_depth)
    plot_data_tas_val$nodes <- NULL
    plot_data_tas_val$Type <- "Validation"
    plot_data_tas_val$Task <- "TAS"
    plot_data_tas$num_nodes <- NULL
    plot_data_tas$Type <- "Test"
    plot_data_tas$Task <- "TAS"
    plot_data <- rbind(plot_data_tes, plot_data_tas, plot_data_tes_val, plot_data_tas_val)
  } else {
    plot_data <- c(plot_data_tes, plot_data_tes_val)
  }

  if (task == "BD") {
    pars_list <- c("lambda", "mu", "Depth", "Model", "Task", "Type")
    data_var_list <- c("lambda", "mu", "lambda_a_diff", "mu_a_diff", "Depth", "Model", "Task", "Type")
  } else if (task == "DDD") {
    pars_list <- c("lambda", "mu", "cap", "Depth", "Model", "Task", "Type")
    data_var_list <- c("lambda", "mu", "cap", "lambda_a_diff", "mu_a_diff", "cap_a_diff", "Depth", "Model", "Task", "Type")
    plot_data$cap <- plot_data$cap / 1000 # scale cap for better representation
    plot_data$cap_a_diff <- plot_data$cap_a_diff / 1000
  } else if (task == "PBD") {

  }

  plot_data <- plot_data %>%
    dplyr::select(dplyr::all_of(data_var_list)) %>%
    tidyr::gather("Parameter", "Value", -dplyr::all_of(pars_list))

  plot_list <- list()

  index <- 1

  for (i in unique(plot_data$Task)) {
    p <- ggplot2::ggplot(plot_data) +
      ggplot2::facet_grid(Depth ~ Parameter,
                          labeller =
                            ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed),
                                              Depth = ggplot2::as_labeller(~paste0(.x, "~'Layer(s)'"), ggplot2::label_parsed)),
                          scales = "free_y") +
      ggplot2::geom_point(data = plot_data %>% dplyr::filter(Task == i),
                          ggplot2::aes_string(xvar, "Value", color = "Type", alpha = "Type")) +
      #ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
      #ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
      ggplot2::scale_alpha_manual(name = "Type", values = c(0.5, 0.1)) +
      viridis::scale_color_viridis(discrete = T) +
      ggplot2::theme(legend.position = "none",
                     plot.background = ggplot2::element_blank(),
                     panel.background = ggplot2::element_blank())

    if (length(unique(plot_data$Task)) == 1) {
      p <- p + ggplot2::labs(x = xlabel, y = "Absolute difference")
    } else {
      if (index == 1) {
        p <- p + ggplot2::labs(x = NULL, y = "Absolute difference")
      } else {
        p <- p + ggplot2::labs(x = xlabel, y = NULL)
      }
    }

    if (task == "BD") {
      p <- p +  ggplot2::coord_cartesian(ylim = c(-0.25, 0.25))
    } else if (task == "DDD") {
      p <- p +  ggplot2::coord_cartesian(ylim = c(-0.6, 0.6))
    }

    if (grepl("TES", i)) {
      p <- p + ggplot2::ggtitle(paste0("GNN ", " Extant trees"))
    } else if (grepl("TAS", i)) {
      p <- p + ggplot2::ggtitle(paste0("GNN ", " Full trees"))
    }

    p <- p +
      ggplot2::geom_vline(xintercept = within_range[1], linetype = "dashed", color = "blue") +
      ggplot2::geom_vline(xintercept = within_range[2], linetype = "dashed", color = "blue")

    plot_list[[length(plot_list) + 1]] <- p
    index <- index + 1
  }

  out <- patchwork::wrap_plots(plotlist = plot_list, ncol = length(plot_list)) +
    patchwork::plot_annotation(title = plot_title)

  return(out)
}


#' @export plot_scatter_difference_in_out_sample_abs_by_layer
plot_scatter_difference_in_out_sample_abs_by_layer <- function(path = NULL,
                                                      task = "BD",
                                                      model = "diffpool",
                                                      depth = 2) {
  config_name <- paste0(tolower(task), "_", "sim.yaml")
  within_range <- yaml::read_yaml(file.path("Config", config_name))$within_ranges

  xvar_list <- NULL
  xlabel_list <- NULL
  if (task == "BD") {
    xvar_list <- c("lambda", "mu", "(lambda - mu)")
    xlabel_list <- c("Speciation rate", "Extinction rate", "Net speciation rate")
  }
  if (task == "DDD") {
    xvar_list <- c("lambda", "mu", "cap")
    xlabel_list <- c("Speciation rate", "Extinction rate", "Carrying capacity")
  }
  if (task == "PBD") {
    xvar_list <- c("lambda1", "lambda2", "lambda3", "mu1", "mu2")
    xlabel_list <- c("Speciation rate of good species", "Speciation completion rate", "Speciation rate of incipient species", "Extinction rate of good species", "Extinction rate of incipient species")
  }
  if (task == "EVE") {
    xvar_list <- c("lambda", "mu", "beta_n", "beta_phi")
    xlabel_list <- c("Speciation rate", "Extinction rate", "Species richness effect", "Evolutionary relatedness effect")
  }

  plot_title <- NULL
  if (task == "BD") {
    plot_title <- "Birth-Death Model"
  } else if (task == "DDD") {
    plot_title <- "Diversity-Dependent-Diversification Model"
  } else if (task == "PBD") {
    plot_title <- "Protracted Birth-Death Model"
  } else if (task == "EVE") {
    plot_title <- "Evolutionary-Relatedness-Dependent Model"
  }

  plot_data_tes_val <- load_final_difference_by_layer(path = path, task_type = paste0(task, "_VAL_TES"), model_type = model, depth = depth)
  plot_data_tes <- load_final_difference_by_layer(path = path, task_type = paste0(task, "_FREE_TES"), model_type = model, depth = depth)
  plot_data_tes_val <- plot_data_tes_val %>% dplyr::slice_sample(n = nrow(plot_data_tes), replace = TRUE)
  plot_data_tes_val$nodes <- NULL
  plot_data_tes_val$Type <- "Validation"
  plot_data_tes_val$Task <- "TES"
  plot_data_tes$num_nodes <- NULL
  plot_data_tes$Type <- "Test"
  plot_data_tes$Task <- "TES"

  plot_data_tas_val <- NULL
  plot_data <- NULL
  if (task != "PBD") {
    plot_data_tas_val <- load_final_difference_by_layer(path = path, task_type = paste0(task, "_VAL_TAS"), model_type = model, depth = depth)
    plot_data_tas <- load_final_difference_by_layer(path = path, task_type = paste0(task, "_FREE_TAS"), model_type = model, depth = depth)
    plot_data_tas_val <- plot_data_tas_val %>% dplyr::slice_sample(n = nrow(plot_data_tas), replace = TRUE)
    plot_data_tas_val$nodes <- NULL
    plot_data_tas_val$Type <- "Validation"
    plot_data_tas_val$Task <- "TAS"
    plot_data_tas$num_nodes <- NULL
    plot_data_tas$Type <- "Test"
    plot_data_tas$Task <- "TAS"
    plot_data <- rbind(plot_data_tes, plot_data_tas, plot_data_tes_val, plot_data_tas_val)
  } else {
    plot_data <- rbind(plot_data_tes, plot_data_tes_val)
  }

  if (task == "BD") {
    pars_list <- c("lambda", "mu", "Model", "Task", "Type")
    data_var_list <- c("lambda", "mu", "lambda_a_diff", "mu_a_diff", "Model", "Task", "Type")
  } else if (task == "DDD") {
    pars_list <- c("lambda", "mu", "cap", "Model", "Task", "Type")
    data_var_list <- c("lambda", "mu", "cap", "lambda_a_diff", "mu_a_diff", "cap_a_diff", "Model", "Task", "Type")
    plot_data$cap_a_diff <- plot_data$cap_a_diff / 1000
  } else if (task == "PBD") {
    pars_list <- c("lambda1","lambda2", "lambda3", "mu1", "mu2", "Model", "Task", "Type")
    data_var_list <- c("lambda1","lambda2", "lambda3", "mu1", "mu2",
                   "lambda1_a_diff", "lambda2_a_diff", "lambda3_a_diff", "mu1_a_diff", "mu2_a_diff",
                   "Model", "Task", "Type")
  }

  plot_data <- plot_data %>%
    dplyr::select(dplyr::all_of(data_var_list)) %>%
    tidyr::gather("Parameter", "Value", -dplyr::all_of(pars_list))

  plot_list <- list()

  for (i in unique(plot_data$Task)) {
    plot_sub_list <- list()
    index <- 1
    for (j in seq_len(length(xvar_list))) {
       p <- ggplot2::ggplot(plot_data) +
            ggplot2::facet_wrap(. ~ Parameter,
                                labeller =
                                  ggplot2::labeller(Parameter = ggplot2::as_labeller(~difference_var_to_label(.x), ggplot2::label_parsed)),
                                scales = "free_y", nrow = 1) +
            ggplot2::geom_point(data = plot_data %>% dplyr::filter(Task == i),
                                ggplot2::aes_string(xvar_list[j], "Value", color = "Type", alpha = "Type")) +
            #ggplot2::geom_hline(yintercept = abline_range[1], linetype = "dashed", color = abline_color) +
            #ggplot2::geom_hline(yintercept = abline_range[2], linetype = "dashed", color = abline_color) +
            nord::scale_color_nord(palette = "frost", discrete = T) +
            ggplot2::scale_alpha_manual(name = "Type", values = c(0.5, 0.09)) +
            ggplot2::labs(x = xlabel_list[j], y = NULL) +
            ggplot2::theme(legend.position = "none",
                           plot.background = ggplot2::element_blank(),
                           panel.background = ggplot2::element_blank())

      if (xvar_list[j] != "(lambda - mu)") {
        if (xvar_list[j] == "lambda") {
          p <- p +
            ggplot2::geom_vline(xintercept = within_range[[1]][1], linetype = "dashed", color = "blue") +
            ggplot2::geom_vline(xintercept = within_range[[1]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "mu") {
          p <- p +
              ggplot2::geom_vline(xintercept = within_range[[2]][1], linetype = "dashed", color = "blue") +
              ggplot2::geom_vline(xintercept = within_range[[2]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "cap") {
          p <- p +
              ggplot2::geom_vline(xintercept = within_range[[3]][1], linetype = "dashed", color = "blue") +
              ggplot2::geom_vline(xintercept = within_range[[3]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "lambda1") {
          p <- p +
            ggplot2::geom_vline(xintercept = within_range[[1]][1], linetype = "dashed", color = "blue") +
            ggplot2::geom_vline(xintercept = within_range[[1]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "lambda2") {
          p <- p +
            ggplot2::geom_vline(xintercept = within_range[[2]][1], linetype = "dashed", color = "blue") +
            ggplot2::geom_vline(xintercept = within_range[[2]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "lambda3") {
          p <- p +
            ggplot2::geom_vline(xintercept = within_range[[3]][1], linetype = "dashed", color = "blue") +
            ggplot2::geom_vline(xintercept = within_range[[3]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "mu1") {
          p <- p +
            ggplot2::geom_vline(xintercept = within_range[[4]][1], linetype = "dashed", color = "blue") +
            ggplot2::geom_vline(xintercept = within_range[[4]][2], linetype = "dashed", color = "blue")
        } else if (xvar_list[j] == "mu2") {
          p <- p +
            ggplot2::geom_vline(xintercept = within_range[[5]][1], linetype = "dashed", color = "blue") +
            ggplot2::geom_vline(xintercept = within_range[[5]][2], linetype = "dashed", color = "blue")
        }
      }

      if (task == "BD") {
        p <- p +  ggplot2::coord_cartesian(ylim = c(-0.35, 0.35))
      } else if (task == "DDD") {
        p <- p +  ggplot2::coord_cartesian(ylim = c(-0.6, 0.6))
      } else if (task == "PBD") {
        p <- p +  ggplot2::coord_cartesian(ylim = c(-0.5, 0.5))
      }

      if (index == 1) {
        if (grepl("TES", i)) {
          p <- p + ggplot2::ggtitle(paste0("GNN ", " Extant trees"))
        } else if (grepl("TAS", i)) {
          p <- p + ggplot2::ggtitle(paste0("GNN ", " Full trees"))
        }
      }

      index <- index + 1
      plot_sub_list[[length(plot_sub_list) + 1]] <- p
    }

    plot_list[[length(plot_list) + 1]] <- patchwork::wrap_plots(plotlist = plot_sub_list,
                                                                ncol = 1,
                                                                byrow = FALSE,
                                                                guides = "collect")
  }



  out <- patchwork::wrap_plots(plotlist = plot_list, ncol = length(unique(plot_data$Task)), byrow = FALSE) +
    patchwork::plot_annotation(title = paste0(plot_title, " (Absolute Difference)"))

  return(out)
}