# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(patchwork)
library(pheatmap)

# Define theme settings
pca_theme <- theme_bw() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 20, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 20, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 20),
    axis.text.x = element_text(color = "black", size = 20)
  )

# Define color palettes
group_colors <- c(
  "health" = "grey",
  "met" = "#EFC260",
  "mast" = "#75C2DC"
)

heatmap_colors <- colorRampPalette(c("white", "#bfd8bd", "#f28482", "#c1121f"))(10)

# Function to load and preprocess spectral data
load_spectral_data <- function(file_path) {
  # Load spectral data
  spc <- read.csv(file_path, sep = ",", header = TRUE, row.names = 1)
  names(spc) <- gsub("^X", "", names(spc))
  
  # Define wave number ranges
  cols <- colnames(spc)
  keep_cols <- cols[!is.na(cols) & 
                     cols >= 1000 & cols <= 3000 & 
                     !(cols >= 1580 & cols <= 1700) &
                     !(cols >= 1800 & cols <= 2800)]
  
  # Filter data for different groups
  health <- spc %>% filter(disease == 0 & dim <= 7) %>% filter(dim != 0)
  health <- health[, keep_cols]
  
  met <- spc[!is.na(spc$dim_met) & is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
               is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ] %>% 
    filter(dim != 0)
  met <- met[, keep_cols]
  
  mast <- spc[is.na(spc$dim_met) & !is.na(spc$dim_mast) & is.na(spc$dim_ket) & 
                is.na(spc$dim_da) & spc$dim <= 7 & spc$disease_in <= 7, ] %>% 
    filter(dim != 0)
  mast <- mast[, keep_cols]
  
  return(list(
    health = health,
    met = met,
    mast = mast,
    keep_cols = keep_cols
  ))
}

# Function to sample data for heatmap
sample_data_for_heatmap <- function(data, mins = 30) {
  health_sub <- data$health[sample(nrow(data$health), mins), ]
  met_sub <- data$met[sample(nrow(data$met), mins), ]
  mast_sub <- data$mast[sample(nrow(data$mast), mins), ]
  
  # Combine data
  df <- rbind(health_sub, met_sub, mast_sub) %>% as.data.frame()
  df <- df[, rev(names(df))]
  
  # Create annotation data
  annotation_row = data.frame(
    group = c(rep('health', mins), rep('met', mins), rep('mast', mins))
  )
  rownames(annotation_row) = rownames(df)
  
  ann_colors = list(
    group = group_colors
  )
  
  return(list(
    data = df,
    annotation = annotation_row,
    colors = ann_colors
  ))
}

# Function to count wave numbers in specific ranges
count_wave_numbers <- function(data, min_range, max_range) {
  count <- sum(sapply(colnames(data), function(x) {
    num <- as.numeric(x)
    !is.na(num) && num >= min_range && num <= max_range
  }))
  return(count)
}

# Function to create heatmap
create_heatmap <- function(data, gaps_col = c(52, 78), show_legend = TRUE) {
  options(repr.plot.width = 14, repr.plot.height = 7)
  
  p <- pheatmap(
    data$data, 
    gaps_col = gaps_col, 
    clustering_distance_rows = "euclidean", 
    clustering_method = "complete",
    fontsize_col = 1, 
    fontsize_row = 1, 
    show_rownames = TRUE, 
    cellwidth = 3.5, 
    cellheight = 4.5, 
    border_color = "white", 
    color = heatmap_colors, 
    cluster_rows = TRUE, 
    cluster_cols = FALSE, 
    treeheight_row = 0, 
    annotation_row = data$annotation, 
    annotation_colors = data$colors,
    annotation_legend = show_legend,
    legend = show_legend
  )
  
  return(p)
}

# Function to extract data for specific wave number ranges
extract_wave_range_data <- function(spc_data, min_range, max_range) {
  cols <- colnames(spc_data)
  keep_cols <- cols[!is.na(cols) & cols >= min_range & cols <= max_range]
  
  health <- spc_data$health[, keep_cols]
  met <- spc_data$met[, keep_cols]
  mast <- spc_data$mast[, keep_cols]
  
  health$group <- "health"
  met$group <- "met"
  mast$group <- "mast"
  
  combined_data <- rbind(health, met, mast)
  
  return(list(
    data = combined_data,
    numeric_data = combined_data %>% select(-group)
  ))
}

# Function to perform PCA and create plot
create_pca_plot <- function(data) {
  # Perform PCA
  pca_result <- prcomp(data$numeric_data, scale. = TRUE)
  
  # Extract first two principal components
  pca_df <- data.frame(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2],
    group = data$data$group
  )
  
  # Calculate variance explained
  var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
  pc1_var <- round(var_explained[1] * 100, 1)
  pc2_var <- round(var_explained[2] * 100, 1)
  
  # Create scatter plot
  options(repr.plot.width = 4, repr.plot.height = 3.5)
  
  p <- ggplot(pca_df, aes(x = PC1, y = PC2)) +
    geom_point(aes(color = group), size = 3.5) +
    stat_ellipse(aes(color = group), type = "norm", level = 0.95, linetype = 1, lwd = 2) +  
    scale_color_manual(values = group_colors) +
    pca_theme + 
    labs(
      x = paste0("PC1 (", pc1_var, "%)"),
      y = paste0("PC2 (", pc2_var, "%)")
    )
  
  return(p)
}

# Main execution function
main <- function() {
  # Load spectral data
  spectral_data <- load_spectral_data('./spc.csv')
  
  # Count wave numbers in specific ranges
  r1_count <- count_wave_numbers(spectral_data$health, 2800, 3000)
  r2_count <- count_wave_numbers(spectral_data$health, 1700, 1800)
  r3_count <- count_wave_numbers(spectral_data$health, 1000, 1585)
  
  print(paste("R1 count:", r1_count))
  print(paste("R2 count:", r2_count))
  print(paste("R3 count:", r3_count))
  
  # Sample data for heatmap
  heatmap_data <- sample_data_for_heatmap(spectral_data)
  
  # Create heatmap
  heatmap_plot <- create_heatmap(heatmap_data)
  
  # Create heatmap without legend for saving
  heatmap_no_legend <- create_heatmap(heatmap_data, show_legend = FALSE)
  gt <- heatmap_no_legend$gtable
  
  # Extract data for specific wave ranges and create PCA plots
  range1_data <- extract_wave_range_data(spectral_data, 1000, 1580)
  range2_data <- extract_wave_range_data(spectral_data, 1700, 1800)
  range3_data <- extract_wave_range_data(spectral_data, 2800, 3000)
  
  pca_plot1 <- create_pca_plot(range1_data)
  pca_plot2 <- create_pca_plot(range2_data)
  pca_plot3 <- create_pca_plot(range3_data)
  
  # Return results
  return(list(
    heatmap = heatmap_plot,
    heatmap_no_legend = gt,
    pca_plots = list(
      range1 = pca_plot1,
      range2 = pca_plot2,
      range3 = pca_plot3
    )
  ))
}

# Run the script
results <- main()


