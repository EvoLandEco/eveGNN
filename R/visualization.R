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
                   plot.margin = ggplot2::unit(c(5, 5, 5, 5), "pt"))

  return(patch)
}