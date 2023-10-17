#' @export plot_model_performance
plot_model_performance <- function(data, group_var = stop("Grouping variable not provided."), legend_title = NULL, accuracy_ref = NULL, loss_ref = NULL) {
  loss_plot <- ggplot2::ggplot(data) +
    ggplot2::geom_line(ggplot2::aes(x = Epoch, y = Loss, color = eval(parse(text = group_var)))) +
    ggplot2::geom_hline(yintercept = loss_ref, linetype = "dashed", color = "black") +
    #ggplot2::ylim(0, loss_ref) +
    ggplot2::annotate("text",
                      x = median(data$Epoch),
                      y = loss_ref * 1.05,
                      label = paste0("Loss = ", loss_ref),
                      color = "black", size = 3) +
    ggplot2::guides(color = ggplot2::guide_legend(nrow = 2)) +
    ggplot2::labs(y = "Loss", color = legend_title) +
    ggplot2::theme(aspect.ratio = 5 / 6)

  accu_plot <- ggplot2::ggplot(data) +
    ggplot2::geom_line(ggplot2::aes(x = Epoch, y = Train_Accuracy, color = eval(parse(text = group_var)))) +
    ggplot2::geom_hline(yintercept = accuracy_ref, linetype = "dashed", color = "black") +
    #ggplot2::ylim(accuracy_ref, 1) +
    ggplot2::annotate("text",
                      x = median(data$Epoch),
                      y = accuracy_ref * 1.05,
                      label = paste0("Accuracy = ", accuracy_ref),
                      color = "black",
                      size = 3) +
    ggplot2::guides(color = ggplot2::guide_legend(nrow = 2)) +
    ggplot2::labs(y = "Accuracy", color = legend_title) +
    ggplot2::theme(aspect.ratio = 5 / 6)

  patch <- loss_plot +
    accu_plot +
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