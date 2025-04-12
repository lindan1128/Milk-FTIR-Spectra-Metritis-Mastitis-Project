# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(patchwork)
library(scales)

# Define theme settings
spectral_plot_theme <- theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 20),
    axis.text.x = element_text(color = "black", size = 0)
  )

# Define color palette
group_colors <- c(
  "met" = "#EFC260",
  "mast" = "#75C2DC"
)

# Function to load and preprocess spectral data
load_spectral_data <- function(file_path) {
  # Load spectral data
  spc <- read.csv(file_path, sep = ",", header = TRUE, row.names = 1)
  names(spc) <- gsub("^X", "", names(spc))
  
  # Define columns to keep (wavenumber range)
  cols <- colnames(spc)
  keep_cols <- cols[!is.na(cols) & 
                     cols >= 1000 & cols <= 3000 & 
                     !(cols >= 1580 & cols <= 1700) & 
                     !(cols >= 1800 & cols <= 2800) | cols == 'cow_id']
  
  return(list(
    spc = spc,
    keep_cols = keep_cols
  ))
}

# Function to filter data for specific groups and days
filter_data <- function(data, group_type, day) {
  if (group_type == "met") {
    filtered <- data[!is.na(data$dim_met) & is.na(data$dim_mast) & 
                      is.na(data$dim_ket) & is.na(data$dim_da) & 
                      data$dim <= 7 & data$disease_in <= 7, ]
  } else if (group_type == "mast") {
    filtered <- data[!is.na(data$dim_mast) & is.na(data$dim_met) & 
                      is.na(data$dim_ket) & is.na(data$dim_da) & 
                      data$dim <= 7 & data$disease_in <= 7, ]
  } else if (group_type == "health") {
    filtered <- data %>% filter(disease == 0 & dim <= 7)
  }
  
  # Filter by day if specified
  if (!is.null(day)) {
    filtered <- filtered %>% filter(dim == day)
  }
  
  return(filtered)
}

# Function to process spectral data for a specific group and day
process_spectral_data <- function(data, keep_cols, group_type, day) {
  # Filter data
  filtered_data <- filter_data(data, group_type, day)
  
  # Extract relevant columns
  group_sub <- filtered_data[, keep_cols]
  
  # Calculate maximum absorbance
  group_sub$max <- apply(group_sub[, 1:(length(keep_cols)-1)], 1, max)
  
  # Split into high and low absorbance groups
  group_sub_l <- group_sub %>% filter(max < 0.1) %>%
    dplyr::select(-max)
  
  group_sub_h <- group_sub %>% filter(max > 0.1) %>%
    dplyr::select(-max)
  
  # Reshape data for plotting
  group_sub_l <- gather(group_sub_l, key = "wavenum", value = "value", -cow_id)
  group_sub_l$wavenum <- as.numeric(group_sub_l$wavenum)
  group_sub_l$group <- 'l'
  
  group_sub_h <- gather(group_sub_h, key = "wavenum", value = "value", -cow_id)
  group_sub_h$wavenum <- as.numeric(group_sub_h$wavenum)
  group_sub_h$group <- 'h'
  
  # Combine high and low groups
  combined_data <- rbind(group_sub_h, group_sub_l)
  
  return(combined_data)
}

# Function to create spectral plot
create_spectral_plot <- function(data, group_color) {
  p <- ggplot(data, aes(wavenum, value, group = cow_id)) +   
    geom_line(color = group_color, size = 2) +
    facet_wrap(. ~ group) +
    scale_x_reverse() +
    ylim(-0.1, 1) +
    scale_y_continuous(limits = c(-0.1, 1),
                      labels = scales::number_format(accuracy = 0.1)) +
    spectral_plot_theme
  
  return(p)
}

# Main execution function
main <- function() {
  # Load spectral data
  data_list <- load_spectral_data('./spc.csv')
  spc <- data_list$spc
  keep_cols <- data_list$keep_cols
  
  # Process data for met group (day 0)
  met_data <- process_spectral_data(spc, keep_cols, "met", 0)
  
  # Create plot for met group
  met_plot <- create_spectral_plot(met_data, group_colors["met"])
  
  # Process data for mast group (day 0)
  mast_data <- process_spectral_data(spc, keep_cols, "mast", 0)
  
  # Create plot for mast group
  mast_plot <- create_spectral_plot(mast_data, group_colors["mast"])
  
  # Set plot dimensions
  options(repr.plot.width = 9, repr.plot.height = 2.7)
  
  # Return plots
  return(list(
    met_plot = met_plot,
    mast_plot = mast_plot
  ))
}

# Run the script
results <- main()

# Display plots
print(results$met_plot)
print(results$mast_plot)
