# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(patchwork)
library(scales)

# Define theme settings
bar_plot_theme <- theme_bw() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 12),
    axis.text.x = element_text(color = "black", size = 0)
  )

box_plot_theme <- theme_bw() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 18),
    axis.text.x = element_text(color = "black", size = 0)
  )

# Define color palette
model_colors <- c(
  "pls-da" = "grey70",
  "rf" = "#EFC260",
  "lstm" = "#75C2DC"
)

inner_colors <- c(
  "outer" = "grey70",
  "inner" = "grey90",
  "inner_t" = "#4cc9f0"
)

# Function to load model data
load_model_data <- function(model_name, comparison_type, file_type) {
  file_path <- paste0('./', model_name, '_2/', comparison_type, '/', file_type, '.csv')
  data <- read.csv(file_path, sep = ",", header = TRUE, row.names = NULL)
  data$model <- model_name
  return(data)
}

# Function to load all model data
load_all_model_data <- function(file_type) {
  # Define models and comparison types
  models <- c('pls-da', 'rf', 'lstm')
  comparisons <- c('health_vs_met', 'health_vs_mast', 'mast_vs_met')
  
  # Initialize empty list to store data
  all_data <- list()
  
  # Load data for each model and comparison
  for (model in models) {
    for (comparison in comparisons) {
      data <- load_model_data(model, comparison, file_type)
      all_data[[paste(model, comparison, sep = "_")]] <- data
    }
  }
  
  # Combine all data
  combined_data <- do.call(rbind, all_data)
  return(combined_data)
}

# Function to create bar plot
create_bar_plot <- function(data, metric, y_limits = c(0, 1)) {
  p <- ggplot(data, aes(x = dim, y = get(paste0("outer_", metric, "_mean")), fill = model)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9), width = 0.84, alpha = 0.8) +
    geom_errorbar(
      aes(ymin = get(paste0("outer_", metric, "_ci_low")), 
          ymax = get(paste0("outer_", metric, "_ci_up"))),
      position = position_dodge(width = 0.9),
      width = 0.2,
      size = 0.8,
      color = '#adb5bd'
    ) +
    scale_y_continuous(limits = y_limits,
                      labels = scales::number_format(accuracy = 0.1)) +
    scale_fill_manual(values = model_colors) +
    bar_plot_theme
  
  return(p)
}

# Function to create box plot
create_box_plot <- function(data, model_name, y_limits = c(0.25, 1)) {
  # Prepare data
  plot_data <- subset(data, select = c(dim, outer_auc_mean, inner_val_auc_mean, inner_train_auc_mean))
  plot_data <- plot_data %>%
    pivot_longer(
      cols = c(outer_auc_mean, inner_val_auc_mean, inner_train_auc_mean),
      names_to = "group",
      values_to = "auc"
    ) %>%
    mutate(
      group = case_when(
        group == "outer_auc_mean" ~ "outer",
        group == "inner_val_auc_mean" ~ "inner"
      )
    )
  plot_data <- na.omit(plot_data)
  plot_data$group <- factor(plot_data$group, levels = c("outer", "inner", "inner_t"))
  
  # Create plot
  p <- ggplot(plot_data, aes(x = group, y = auc, fill = group)) +
    geom_boxplot(alpha = 0.5, width = 0.5, notch = FALSE) +
    geom_point() +
    geom_line(aes(group = dim)) +
    scale_fill_manual(values = inner_colors) +
    ylim(y_limits[1], y_limits[2]) +
    box_plot_theme
  
  return(p)
}

# Function to calculate improvement ratio
calculate_improvement_ratio <- function(data) {
  improvement <- data %>%
    group_by(dim) %>%
    summarise(
      inner_auc = auc[group == "inner"],
      outer_auc = auc[group == "outer"],
      improvement_ratio = (inner_auc - outer_auc) / inner_auc
    ) %>%
    summarise(
      mean_improvement = mean(improvement_ratio)
    )
  
  return(improvement)
}

# Function to create feature comparison plot
create_feature_plot <- function(data, model_name, comparison_type, y_limits = c(0, 1)) {
  # Filter and prepare data
  plot_data <- data %>% 
    filter(type == comparison_type) %>% 
    filter(model == model_name) %>% 
    filter(dim != 0)
  
  plot_data <- na.omit(plot_data)
  plot_data$feature <- factor(plot_data$feature, 
                             levels = c('spc+my+scc+parity', 'spc+parity', 'spc+scc', 'spc+my', 'spc'))
  
  # Calculate summary statistics
  plot_data <- plot_data %>% 
    group_by(feature) %>% 
    summarise(
      mean = mean(outer_auc_mean, na.rm = TRUE), 
      sd = sd(outer_auc_mean, na.rm = TRUE)
    )
  
  # Create plot
  p <- ggplot(plot_data, aes(x = feature, y = mean)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), 
             alpha = 0.7, width = 0.5, color = 'grey10', 
             fill = model_colors[model_name]) +
    geom_errorbar(
      aes(ymin = mean - sd, ymax = mean + sd),
      position = position_dodge(width = 0.8),
      width = 0.25,
      color = '#adb5bd'
    ) +
    geom_line(aes(group = 1), color = model_colors[model_name], alpha = 1, size = 1.5) +
    geom_point(color = "grey30", size = 2) +
    scale_y_continuous(limits = y_limits,
                      labels = scales::number_format(accuracy = 0.1)) +
    theme_classic() +
    theme(
      legend.position = "right",
      plot.title = element_text(size = 28),
      axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
      axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
      axis.ticks = element_line(color = "black"),
      axis.text.y = element_text(color = "black", size = 0),
      axis.text.x = element_text(color = "black", size = 10)
    ) + 
    coord_flip()
  
  return(p)
}

# Main execution function
main <- function() {
  # Load data
  p_values <- load_all_model_data("p_values")
  pre <- load_all_model_data("pre")
  
  # Create plots for different metrics
  metrics <- c("auc", "acc", "sen", "spc")
  comparison_types <- c("health_vs_met", "health_vs_mast", "mast_vs_met")
  
  # Store all plots
  all_plots <- list()
  
  # Create bar plots for each metric and comparison
  for (metric in metrics) {
    for (comparison in comparison_types) {
      plot_name <- paste0(metric, "_", comparison)
      pre_sub <- pre %>% filter(type == comparison) %>% filter(feature == 'spc')
      
      all_plots[[plot_name]] <- create_bar_plot(pre_sub, metric)
    }
  }
  
  # Create box plots for each model and comparison
  for (model in c("pls-da", "rf", "lstm")) {
    for (comparison in comparison_types) {
      plot_name <- paste0("box_", model, "_", comparison)
      pre_sub <- pre %>% 
        filter(type == comparison) %>% 
        filter(feature == 'spc') %>% 
        filter(model == model)
      
      all_plots[[plot_name]] <- create_box_plot(pre_sub, model)
    }
  }
  
  # Create feature comparison plots
  for (model in c("pls-da", "rf", "lstm")) {
    for (comparison in comparison_types) {
      plot_name <- paste0("feature_", model, "_", comparison)
      
      # Set y-limits based on comparison type
      y_limits <- if (comparison == "health_vs_mast") c(0, 0.7) else c(0, 0.8)
      
      all_plots[[plot_name]] <- create_feature_plot(pre, model, comparison, y_limits)
    }
  }
  
  # Calculate improvement ratios
  improvement_ratios <- list()
  for (model in c("pls-da", "rf", "lstm")) {
    for (comparison in comparison_types) {
      pre_sub <- pre %>% 
        filter(type == comparison) %>% 
        filter(feature == 'spc') %>% 
        filter(model == model)
      
      pre_sub <- subset(pre_sub, select = c(dim, outer_auc_mean, inner_val_auc_mean, inner_train_auc_mean))
      pre_sub <- pre_sub %>%
        pivot_longer(
          cols = c(outer_auc_mean, inner_val_auc_mean, inner_train_auc_mean),
          names_to = "group",
          values_to = "auc"
        ) %>%
        mutate(
          group = case_when(
            group == "outer_auc_mean" ~ "outer",
            group == "inner_val_auc_mean" ~ "inner"
          )
        )
      pre_sub <- na.omit(pre_sub)
      
      improvement_ratios[[paste(model, comparison, sep = "_")]] <- calculate_improvement_ratio(pre_sub)
    }
  }
  
  # Return all results
  return(list(
    plots = all_plots,
    improvement_ratios = improvement_ratios
  ))
}

# Run the script
results <- main()

