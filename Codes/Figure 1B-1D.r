# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(patchwork)
library(alluvial)

# Define theme settings
main_theme <- theme_linedraw() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 28),
    axis.text.x = element_text(color = "black", size = 0)
  )

boxplot_theme <- theme_bw() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 10),
    axis.text.x = element_text(color = "black", size = 0)
  )

# Define color palettes
group_colors <- c(
  "health" = "grey",
  "met" = "#EFC260",
  "mast" = "#75C2DC"
)

# Function to load and preprocess data
load_data <- function(file_path) {
  # Load spectral data
  spc <- read.csv(file_path, sep = ",", header = TRUE, row.names = 1)
  names(spc) <- gsub("^X", "", names(spc))
  
  # Filter data for different groups
  met <- spc[!is.na(spc$dim_met) & is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
              is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ]
  mast <- spc[is.na(spc$dim_met) & !is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
               is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ]
  health <- spc %>% filter(disease == 0 & dim <= 7)
  
  # Process met data
  met$parity <- sapply(met$parity, function(x) ifelse(x >= 3, 3, x))
  met <- subset(met, select = c('parity', 'dim', 'disease_in'))
  colnames(met) <- c('parity', 'dim', 'disease_in')
  met$group <- 'met'
  met <- met %>%
    group_by(disease_in, dim, parity, group) %>% 
    summarise(n = n())
  
  # Process mast data
  mast$parity <- sapply(mast$parity, function(x) ifelse(x >= 3, 3, x))
  mast <- subset(mast, select = c('parity', 'dim', 'disease_in'))
  colnames(mast) <- c('parity', 'dim', 'disease_in')
  mast$group <- 'mast'
  mast <- mast %>%
    group_by(disease_in, dim, parity, group) %>% 
    summarise(n = n())
  
  # Combine data
  met_mast <- rbind(met, mast)
  met_mast <- met_mast[, c('group', 'parity', 'dim', 'disease_in', 'n')]
  
  # Process combined data for violin plots
  health$group <- 'health'
  met_full <- spc[!is.na(spc$dim_met) & is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
                   is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ]
  met_full$group <- 'met'
  mast_full <- spc[is.na(spc$dim_met) & !is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
                    is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ]
  mast_full$group <- 'mast'
  
  all <- rbind(health, met_full, mast_full)
  all$group <- factor(all$group, levels = c("health", "met", "mast"))
  
  return(list(
    met_mast = met_mast,
    all = all
  ))
}

# Function to create alluvial plot
create_alluvial_plot <- function(data) {
  options(repr.plot.width = 9, repr.plot.height = 2.6)
  
  alluvial(data[, 1:4], 
           freq = data$n,
           col = ifelse(data$group == "met", "#EFC260", "#75C2DC"),
           border = ifelse(data$group == "met", "#EFC260", "#75C2DC"),
           gap.width = 0.2,
           alpha = 0.0,
           cex = 1,
           cw = 0.25)
}

# Function to create violin plot
create_violin_plot <- function(data, y_var, title = "") {
  ggplot(data, aes_string("group", y_var, fill = "group")) + 
    geom_violin(alpha = 0.7) +
    geom_boxplot(width = 0.3, fill = "white") +
    stat_summary(fun.y = mean, geom = "point", shape = 23, size = 2) + 
    scale_fill_manual(values = group_colors) +
    main_theme +
    ggtitle(title)
}

# Function to create boxplot with trend lines
create_boxplot_with_trend <- function(data, y_var) {
  data$dim <- as.factor(data$dim)
  
  ggplot(data, aes_string(x = "dim", y = y_var, fill = "group")) + 
    geom_boxplot(width = 0.5, alpha = 0.7) +
    stat_summary(fun.y = median, geom = 'line', 
                aes(group = group, color = group), 
                position = position_dodge(width = 0.9), size = 1) +
    scale_fill_manual(values = group_colors) +
    scale_color_manual(values = group_colors) +
    boxplot_theme
}

# Function to perform statistical tests
perform_statistical_tests <- function(data, var_name, dim_value) {
  data_sub <- data %>% filter(dim == dim_value)
  
  # Perform pairwise Wilcoxon test with both corrected and uncorrected p-values
  corrected <- as.data.frame(pairwise.wilcox.test(data_sub[[var_name]], 
                                                 data_sub$group, 
                                                 p.adjust.method = 'BH')$p.value)
  uncorrected <- as.data.frame(pairwise.wilcox.test(data_sub[[var_name]], 
                                                   data_sub$group, 
                                                   p.adjust.method = 'none')$p.value)
  
  # Format p-values
  df_uncorrected <- corrected
  df_uncorrected[] <- as.data.frame(
    lapply(uncorrected, function(x) sprintf("%.4f", x))
  )
  df_corrected <- corrected
  df_corrected[] <- as.data.frame(
    lapply(corrected, function(x) sprintf("%.4f", x))
  )
  
  # Combine corrected and uncorrected p-values
  result <- as.data.frame(
    mapply(function(x, y) paste(x, y, sep = "/"), 
           df_uncorrected, df_corrected)
  )
  
  return(result)
}

# Main execution function
main <- function() {
  # Load and process data
  data <- load_data('./spc.csv')
  
  # Create alluvial plot
  create_alluvial_plot(data$met_mast)
  
  # Create violin plots
  p1 <- create_violin_plot(data$all, "parity", "Parity Distribution")
  p2 <- create_violin_plot(data$all, "milkweightlbs", "Milk Weight Distribution")
  p3 <- create_violin_plot(data$all, "log2(cells)", "Cell Count Distribution")
  
  # Create boxplots with trend lines
  p4 <- create_boxplot_with_trend(data$all, "milkweightlbs")
  p5 <- create_boxplot_with_trend(data$all, "log2(cells)")
  
  # Perform statistical tests for each day
  milk_weight_tests <- lapply(0:7, function(dim) {
    perform_statistical_tests(data$all, "milkweightlbs", dim)
  })
  
  cell_count_tests <- lapply(0:7, function(dim) {
    perform_statistical_tests(data$all, "cells", dim)
  })
  
  # Combine test results
  milk_weight_results <- do.call(rbind, milk_weight_tests)
  cell_count_results <- do.call(rbind, cell_count_tests)
  
  # Save plots
  options(repr.plot.width = 4, repr.plot.height = 8)
  p2_adjusted <- p2 + theme(plot.margin = margin(b = 30, unit = "pt"))
  p3_adjusted <- p3 + theme(plot.margin = margin(t = 30, unit = "pt"))
  combined_plot <- p2_adjusted + p3_adjusted + plot_layout(nrow = 2)
  
  # Return results
  return(list(
    plots = list(
      alluvial = create_alluvial_plot(data$met_mast),
      violin = list(p1, p2, p3),
      boxplot = list(p4, p5),
      combined = combined_plot
    ),
    tests = list(
      milk_weight = milk_weight_results,
      cell_count = cell_count_results
    )
  ))
}

# Run the script
results <- main()

# Print statistical test results
print("Pairwise Wilcoxon tests for parity:")
print(pairwise.wilcox.test(results$all$parity, results$all$group, p.adjust.method = 'BH'))

print("Pairwise Wilcoxon tests for milk weight:")
print(pairwise.wilcox.test(results$all$milkweightlbs, results$all$group, p.adjust.method = 'BH'))

print("Pairwise Wilcoxon tests for cell count:")
print(pairwise.wilcox.test(results$all$cells, results$all$group, p.adjust.method = 'BH'))


