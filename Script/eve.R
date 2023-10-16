library(devtools)
library(roxygen2)
library(ggplot2)
library(patchwork)

ggthemr::ggthemr("flat", layout = "minimal")

detach("package:eveGNN", unload=TRUE)
install_github("EvoLandEco/eveGNN", dependencies = FALSE)

test_control <- readRDS("D:/Habrok/Data/14102023/Data/qt2/10_dsce2_0.6_0.1_0.0_0.0.rds")
test_no_n <- readRDS("D:/Habrok/Data/14102023/Data/qt2/10_dsce2_0.6_0.1_0.0_-0.04.rds")
test_has_n <- readRDS("D:/Habrok/Data/14102023/Data/qt2/10_dsce2_0.6_0.1_-0.04_-0.04.rds")
test_no_phi <- readRDS("D:/Habrok/Data/14102023/Data/qt2/10_dsce2_0.6_0.1_-0.04_0.0.rds")
test_pos_phi <- readRDS("D:/Habrok/Data/14102023/Data/qt2/10_dsce2_0.6_0.1_-0.04_0.001.rds")

test_control$Group <- "Control"
test_control <- dplyr::bind_rows(data.frame(Epoch = 0, Train_Accuracy = 0, Loss = 1.104, Group = "Control"), test_control)
row.names(test_control) <- NULL
test_no_n$Group <- "NO_N"
test_no_n <- dplyr::bind_rows(data.frame(Epoch = 0, Train_Accuracy = 0, Loss = 1.104, Group = "NO_N"), test_no_n)
row.names(test_control) <- NULL
test_has_n$Group <- "HAS_N"
test_has_n <- dplyr::bind_rows(data.frame(Epoch = 0, Train_Accuracy = 0, Loss = 1.104, Group = "HAS_N"), test_has_n)
row.names(test_control) <- NULL
test_no_phi$Group <- "NO_PHI"
test_no_phi <- dplyr::bind_rows(data.frame(Epoch = 0, Train_Accuracy = 0, Loss = 1.104, Group = "NO_PHI"), test_no_phi)
row.names(test_control) <- NULL
test_pos_phi$Group <- "POS_PHI"
test_pos_phi <- dplyr::bind_rows(data.frame(Epoch = 0, Train_Accuracy = 0, Loss = 1.104, Group = "POS_PHI"), test_pos_phi)
row.names(test_control) <- NULL


combined_test <- rbind(test_no_n, test_has_n, test_no_phi, test_control, test_pos_phi)

p_loss <- ggplot(combined_test) +
  geom_line(aes(x = Epoch, y = Loss, color = Group)) +
  geom_hline(yintercept = 1.104, linetype = "dashed", color = "black") +
  ylim(0, 1.2) +
  scale_color_discrete(labels = c("Control",
                                  "Negative N, Negative Phi",
                                  "No N, Negative Phi",
                                  "Negative N, No Phi",
                                  "Negative N, Positive Phi")) +
  #geom_text(aes(x = 100), y = 1.2, label = "Loss = 1.104", color = "black", size = 3) +
  annotate("text", x = 100, y = 1.2, label = "Loss = 1.104", color = "black", size = 3) +
  guides(color = guide_legend(nrow = 2)) +
  labs(y = "Loss") +
  theme(aspect.ratio = 5/6)
p_accu <- ggplot(combined_test) +
  geom_line(aes(x = Epoch, y = Train_Accuracy, color = Group)) +
  geom_hline(yintercept = 1/3, linetype = "dashed", color = "black") +
  ylim(0, 1) +
  scale_color_discrete(labels = c("Control",
                                  "Negative N, Negative Phi",
                                  "No N, Negative Phi",
                                  "Negative N, No Phi",
                                  "Negative N, Positive Phi")) +
  #geom_text(aes(x = 100), y = 0.4, label = "Accuracy = 1/3", color = "black", size = 3) +
  annotate("text", x = 100, y = 0.42, label = "Accuracy = 1/3", color = "black", size = 3) +
  guides(color = guide_legend(nrow = 2)) +
  labs(y = "Accuracy") +
  theme(aspect.ratio = 5/6)

patch <- p_loss + p_accu + patchwork::plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = 'bottom',
        panel.background = element_rect(fill = "transparent", color = NA),
        legend.background = element_rect(fill = "transparent", color = NA))

patch + patchwork::plot_annotation(
  title = 'GNN Training Results',
  subtitle = 'Graph neural network classification of eve simulation results'
)