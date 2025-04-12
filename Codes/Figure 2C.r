# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(patchwork)

# Define theme settings
line_plot_theme <- theme_bw() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 20),
    axis.text.x = element_text(color = "black", size = 0)
  )

# Define color palettes
group_colors <- c(
  "health" = "grey",
  "met" = "#EFC260",
  "mast" = "#75C2DC"
)

# Function to load and preprocess spectral data
load_spectral_data <- function(file_path) {
  # Load spectral data
  spc <- read.csv(file_path, sep = ",", header = TRUE, row.names = 1)
  names(spc) <- gsub("^X", "", names(spc))
  
  # Filter data for different groups
  health <- spc %>% filter(disease == 0 & dim <= 7) 
  met <- spc[!is.na(spc$dim_met) & is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
               is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ] 
  mast <- spc[is.na(spc$dim_met) & !is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
                is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ]
  
  return(list(
    health = health,
    met = met,
    mast = mast
  ))
}

# Function to extract data for a specific wave number
extract_wave_data <- function(data, wave_number) {
  health_sub <- data$health[, c(wave_number, 'dim')]
  met_sub <- data$met[, c(wave_number, 'dim')]
  mast_sub <- data$mast[, c(wave_number, 'dim')]
  
  health_sub$group <- "health"
  met_sub$group <- "met"
  mast_sub$group <- "mast"
  
  combined_data <- rbind(health_sub, met_sub, mast_sub)
  
  return(combined_data)
}

# Function to calculate summary statistics
calculate_summary_stats <- function(data, wave_number) {
  summary_data <- data %>%
    group_by(dim, group) %>%
    summarise(
      mean = mean(!!sym(wave_number)),
      sd = sd(!!sym(wave_number)),
      n = n(),
      se = sd / sqrt(n),
      ci_lower = mean - 1.96 * se,
      ci_upper = mean + 1.96 * se
    )
  
  return(summary_data)
}

# Function to create line plot with confidence intervals
create_line_plot <- function(summary_data, wave_number) {
  p <- ggplot(summary_data, aes(x = dim, y = mean, group = group, color = group)) +
    geom_line(size = 3) +
    geom_point(size = 5) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = group), 
                alpha = 0.2, color = NA) +
    scale_fill_manual(values = group_colors) +
    scale_color_manual(values = group_colors) +
    line_plot_theme
  
  return(p)
}

# Function to perform statistical tests
perform_statistical_tests <- function(data, wave_number) {
  # Rename columns for easier access
  colnames(data) <- c('value', 'dim', 'group')
  
  # Initialize list to store results
  p_values <- list()
  
  # Perform tests for each day
  for (day in 0:7) {
    # Filter data for current day
    day_data <- data %>% filter(dim == as.character(day))
    
    # Perform Wilcoxon tests with different adjustment methods
    corrected <- as.data.frame(pairwise.wilcox.test(day_data$value, day_data$group, p.adjust.method='BH')$p.value)
    uncorrected <- as.data.frame(pairwise.wilcox.test(day_data$value, day_data$group, p.adjust.method='none')$p.value)
    
    # Format p-values
    df_uncorrected <- corrected
    df_uncorrected[] <- as.data.frame(
      lapply(uncorrected, function(x) sprintf("%.4f", x))
    )
    
    df_corrected <- corrected
    df_corrected[] <- as.data.frame(
      lapply(corrected, function(x) sprintf("%.4f", x))
    )
    
    # Combine uncorrected and corrected p-values
    p_values[[as.character(day)]] <- as.data.frame(
      mapply(function(x, y) paste(x, y, sep="/"), 
             df_uncorrected, df_corrected)
    )
  }
  
  # Combine all results
  combined_results <- do.call(rbind, p_values)
  
  return(combined_results)
}

# Main execution function
main <- function() {
  # Load spectral data
  spectral_data <- load_spectral_data('./spc.csv')
  
  # Define wave numbers to analyze
  wave_numbers <- c('2923.87', '2854.44', '1747.38', '1157.2', '1542.94', '1041.48')
  
  # Create plots and perform statistical tests for each wave number
  plots <- list()
  p_values <- list()
  
  for (wave_number in wave_numbers) {
    # Extract data for current wave number
    wave_data <- extract_wave_data(spectral_data, wave_number)
    
    # Calculate summary statistics
    summary_data <- calculate_summary_stats(wave_data, wave_number)
    
    # Create line plot
    plots[[wave_number]] <- create_line_plot(summary_data, wave_number)
    
    # Perform statistical tests
    p_values[[wave_number]] <- perform_statistical_tests(wave_data, wave_number)
  }
  
  # Return results
  return(list(
    plots = plots,
    p_values = p_values
  ))
}

# Run the script
results <- main()

