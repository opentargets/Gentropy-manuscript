# Circular Manhattan Plot using circlize library
# Dataset: unique_measurement_count_per_gene.parquet

# Resolve script directory (works both when sourced and run standalone via Rscript)
if (!exists("fig1_dir")) {
  .argv     <- commandArgs(trailingOnly = FALSE)
  .file_arg <- .argv[startsWith(.argv, "--file=")]
  fig1_dir  <- if (length(.file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
  } else {
    tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
  }
}

# Load required libraries
library(circlize)
library(dplyr)
library(arrow)
library(RColorBrewer)
library(png)

# Function to read parquet file
read_parquet_data <- function(file_path) {
  # Read the parquet file
  data <- read_parquet(file_path)
  return(data)
}

# Function to get real chromosome lengths (GRCh38/hg38)
get_chromosome_lengths <- function() {
  # Human chromosome lengths in base pairs (GRCh38/hg38)
  chr_lengths <- data.frame(
    chromosome = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                   "21", "22", "X", "Y", "MT"),
    length = c(248956422, 242193529, 198295559, 190214555, 181538259, 170805979,
               159345973, 145138636, 138394717, 133797422, 135086622, 133275309,
               114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
               58617616, 64444167, 46709983, 50818468, 156040895, 57227415, 16569)
  )
  return(chr_lengths)
}

# Function to prepare data for circular Manhattan plot
prepare_manhattan_data <- function(data) {
  # Select and rename columns for Manhattan plot
  manhattan_data <- data %>%
    select(
      chromosome = chromosome,
      start = start,
      y_value_diseases = uniqueDiseases,
      y_value_therapeutic = uniqueTherapeuticAreas,
      y_value_measurement = uniqueMeasurement
    ) %>%
    # Ensure chromosome is character and start is numeric
    mutate(
      chromosome = as.character(chromosome),
      start = as.numeric(start),
      y_value_diseases = as.numeric(y_value_diseases),
      y_value_therapeutic = as.numeric(y_value_therapeutic),
      y_value_measurement = as.numeric(y_value_measurement)
    ) %>%
    # Keep all rows - no filtering of missing values
    # Sort by chromosome and position
    arrange(chromosome, start)
  
  return(manhattan_data)
}

# Function to add gene labels outside the plot for genes with uniqueDiseases > n
add_gene_labels_outside <- function(manhattan_data, chr_regions) {
  # Initialize container for labeled points used for highlighting
  .labeled_points <<- data.frame(sector = character(), x = numeric(), label = character(), stringsAsFactors = FALSE)
  # Load the full dataset to get approvedSymbol and uniqueDiseases
  full_data <- read.csv(file.path(fig1_dir, "data", "disease_ta_index_pandas.csv"))
  
  # Filter genes with uniqueDiseases > n
  high_disease_genes <- full_data %>%
    filter(uniqueDiseases > 35) %>%
    select(geneId, approvedSymbol, chromosome, start, uniqueDiseases) %>%
    filter(!is.na(approvedSymbol) & approvedSymbol != "")
  
  cat("Found", nrow(high_disease_genes), "genes with uniqueDiseases > 50\n")
  
  if(nrow(high_disease_genes) > 0) {
    # Find matching genes in the plotted data and prepare for circos.labels
    label_data <- data.frame()
    
    for(i in 1:nrow(high_disease_genes)) {
      gene <- high_disease_genes[i, ]
      chr <- as.character(gene$chromosome)
      start_pos <- gene$start
      
      # Find the corresponding point in manhattan_data
      matching_point <- manhattan_data %>%
        filter(chromosome == chr & start == start_pos)
      
      if(nrow(matching_point) > 0) {
        # Add to label data
        label_data <- rbind(label_data, data.frame(
          sector = chr,
          x = start_pos,
          label = gene$approvedSymbol
        ))
      }
    }
    
    if(nrow(label_data) > 0) {
      # Save labeled positions for later highlighting in tracks
      .labeled_points <<- label_data
      # Use circos.labels to add labels outside the plot
      circos.labels(
        sectors = label_data$sector,
        x = label_data$x,
        labels = label_data$label,
        side = "outside",
        col = "#434343",
        cex = 0.8,
        font = 2,
        connection_height = mm_h(3),
        line_lwd = 0.5
      )
    }
  }
}

# Function to create circular Manhattan plot
create_circular_manhattan <- function(data, output_file = NULL,
                                      center_plot = NULL,
                                      center_r = 0.50) {
  
  # If an output file is provided, open a PNG device with fixed size/DPI
  if(!is.null(output_file)) {
    opened <- FALSE
    # Prefer Quartz on macOS (stable text metrics without external deps)
    if(tolower(Sys.info()[["sysname"]]) == "darwin") {
      try({
        suppressWarnings(png(filename = output_file, width = 1800, height = 1800, res = 300, type = "quartz"))
        opened <- TRUE
        cat("Device: quartz PNG\n")
      }, silent = TRUE)
    }
    # Fall back to Cairo if available
    if(!opened && isTRUE(unname(capabilities("cairo")))) {
      try({
        suppressWarnings(png(filename = output_file, width = 1800, height = 1800, res = 300, type = "cairo-png"))
        opened <- TRUE
        cat("Device: cairo-png\n")
      }, silent = TRUE)
    }
    # Final fallback: default PNG
    if(!opened) {
      suppressWarnings(png(filename = output_file, width = 1800, height = 1800, res = 300))
      cat("Device: default PNG\n")
    }
  }

  # Prepare data
  manhattan_data <- prepare_manhattan_data(data)
  
  # Debug: Check what data exists for each chromosome
  cat("\nDebug: Data summary by chromosome:\n")
  chr_summary <- manhattan_data %>%
    group_by(chromosome) %>%
    summarise(
      n_rows = n(),
      n_valid_start = sum(!is.na(start)),
      n_valid_diseases = sum(!is.na(y_value_diseases)),
      n_valid_therapeutic = sum(!is.na(y_value_therapeutic)),
      n_valid_measurement = sum(!is.na(y_value_measurement)),
      .groups = 'drop'
    )
  print(chr_summary)
  
  # Print detailed table of dots per level per chromosome
  cat("\n=== DOTS PER LEVEL PER CHROMOSOME ===\n")
  dots_table <- manhattan_data %>%
    group_by(chromosome) %>%
    summarise(
      `Measurements (Level 1)` = sum(!is.na(y_value_measurement)),
      `Therapeutic Areas (Level 2)` = sum(!is.na(y_value_therapeutic)),
      `Diseases (Level 3)` = sum(!is.na(y_value_diseases)),
      `Total Genes` = n(),
      .groups = 'drop'
    ) %>%
    arrange(match(chromosome, c(as.character(1:22), "X", "Y", "MT", "M")))
  
  # Print the table with nice formatting
  cat("\nChromosome | Measurements | Therapeutic | Diseases | Total Genes\n")
  cat("-----------|--------------|-------------|----------|------------\n")
  for(i in 1:nrow(dots_table)) {
    cat(sprintf("%-10s | %-12d | %-11d | %-8d | %-11d\n", 
                dots_table$chromosome[i],
                dots_table$`Measurements (Level 1)`[i],
                dots_table$`Therapeutic Areas (Level 2)`[i],
                dots_table$`Diseases (Level 3)`[i],
                dots_table$`Total Genes`[i]))
  }
  
  # Print summary statistics
  cat("\n=== SUMMARY STATISTICS ===\n")
  cat("Total genes with measurement data:", sum(dots_table$`Measurements (Level 1)`), "\n")
  cat("Total genes with therapeutic area data:", sum(dots_table$`Therapeutic Areas (Level 2)`), "\n")
  cat("Total genes with disease data:", sum(dots_table$`Diseases (Level 3)`), "\n")
  cat("Total genes in dataset:", sum(dots_table$`Total Genes`), "\n")
  
  # Check for potential chromosome assignment issues
  cat("\n=== CHROMOSOME BOUNDARY CHECK ===\n")
  chr_boundaries <- manhattan_data %>%
    group_by(chromosome) %>%
    summarise(
      min_start = min(start, na.rm = TRUE),
      max_start = max(start, na.rm = TRUE),
      n_genes = n(),
      .groups = 'drop'
    ) %>%
    arrange(match(chromosome, c(as.character(1:22), "X", "Y", "MT", "M")))
  
  cat("Chromosome boundaries:\n")
  for(i in 1:nrow(chr_boundaries)) {
    cat(sprintf("Chr %-3s: %10d to %10d (%d genes)\n", 
                chr_boundaries$chromosome[i],
                chr_boundaries$min_start[i],
                chr_boundaries$max_start[i],
                chr_boundaries$n_genes[i]))
  }
  
  # Check for overlapping chromosome regions (potential misassignment)
  cat("\nChecking for potential chromosome misassignments...\n")
  for(i in 1:(nrow(chr_boundaries)-1)) {
    for(j in (i+1):nrow(chr_boundaries)) {
      chr1 <- chr_boundaries$chromosome[i]
      chr2 <- chr_boundaries$chromosome[j]
      min1 <- chr_boundaries$min_start[i]
      max1 <- chr_boundaries$max_start[i]
      min2 <- chr_boundaries$min_start[j]
      max2 <- chr_boundaries$max_start[j]
      
      # Check for overlap
      if(max1 >= min2 && max2 >= min1) {
        cat("WARNING: Overlapping regions between", chr1, "and", chr2, "\n")
        cat("  ", chr1, ":", min1, "-", max1, "\n")
        cat("  ", chr2, ":", min2, "-", max2, "\n")
      }
    }
  }
  
  # Get unique chromosomes and sort them in proper numeric order
  chromosomes <- unique(manhattan_data$chromosome)
  # Sort chromosomes: numeric first in proper order (1,2,3...22), then X, Y, etc.
  numeric_chr <- chromosomes[grepl("^[0-9]+$", chromosomes)]
  numeric_chr <- as.numeric(numeric_chr)
  numeric_chr <- sort(numeric_chr)  # This will give proper numeric order: 1,2,3,...,22
  other_chr <- sort(chromosomes[!grepl("^[0-9]+$", chromosomes)])
  chromosomes <- c(as.character(numeric_chr), other_chr)
  
  # Create chromosome regions using real chromosome lengths
  chr_lengths <- get_chromosome_lengths()
  
  # Get chromosomes that exist in our data
  data_chromosomes <- unique(manhattan_data$chromosome)
  
  # Create regions using real chromosome lengths
  chr_regions <- chr_lengths %>%
    filter(chromosome %in% data_chromosomes) %>%
    mutate(
      start = 1,  # All chromosomes start at position 1
      end = length  # End at the real chromosome length
    ) %>%
    select(chromosome, start, end) %>%
    # Order chromosomes in the same way as our sorted chromosomes vector
    arrange(match(chromosome, chromosomes))
  
  cat("\nUsing real chromosome lengths:\n")
  for(i in 1:nrow(chr_regions)) {
    cat(sprintf("Chr %-3s: %10d to %10d (length: %d bp)\n", 
                chr_regions$chromosome[i],
                chr_regions$start[i],
                chr_regions$end[i],
                chr_regions$end[i] - chr_regions$start[i] + 1))
  }
  
  # Clear any previous plot
  circos.clear()
  
  # Initialize the circular plot with bigger size
  circos.par(
    track.height = 0.20,
    cell.padding = c(0, 0, 0, 0),
    gap.degree = 2,
    start.degree = 268.5,
    canvas.xlim = c(-1, 1),
    canvas.ylim = c(-1, 1)
  )
  
  # Create gaps between chromosomes with extra space for y-axis labels
  # Since chromosomes are ordered as: 1, 2, 3, ..., 22, X, Y
  gap_vector <- rep(1, length(chromosomes))
  
  # Find the last chromosome position (should be Y)
  last_pos <- length(chromosomes)
  
  # Create larger gap after the last chromosome (between Y and 1) for y-axis labels
  gap_vector[last_pos] <- 5
  
  # Apply the gap configuration
  circos.par(gap.degree = gap_vector)
  
  # Zero out base-R margins so circlize fills the entire figure region
  par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0))

  # Initialize with our custom chromosome regions
  # pty="s" forces a square plot region; without it the quartz PDF device can
  # produce a very slightly non-square figure region which makes the circle elliptical.
  par(pty = "s")
  circos.initialize(
    factors = chr_regions$chromosome,
    xlim = as.matrix(chr_regions[, c("start", "end")])
  )
  

  # Add gene labels outside the plot for genes with uniqueDiseases > 50
  add_gene_labels_outside(manhattan_data, chr_regions) 
  
  # Create the first Manhattan plot track (Unique Diseases)
  y_max_diseases <- max(manhattan_data$y_value_diseases, na.rm = TRUE)
  circos.track(
    ylim = c(0, y_max_diseases),
    bg.border = NA,
    bg.col = "#DBEAF6",  # Disease track background
    track.height = 0.2,  # Higher track for diseases
    panel.fun = function(x, y) {
      chr = get.cell.meta.data("sector.index")
      xlim = get.cell.meta.data("xlim")
      ylim = get.cell.meta.data("ylim")
      
      # Filter data for current chromosome (show all data points)
      chr_data_before_filter <- manhattan_data %>%
        filter(chromosome == chr)
      
      chr_data <- chr_data_before_filter %>%
        filter(start >= xlim[1] & start <= xlim[2])
      
      # Debug: Check for genes outside chromosome boundaries
      if(nrow(chr_data_before_filter) > 0) {
        outside_genes <- chr_data_before_filter %>%
          filter(start < xlim[1] | start > xlim[2])
        if(nrow(outside_genes) > 0) {
          cat("Debug: Chromosome", chr, "-", nrow(outside_genes), "genes outside boundaries (", 
              nrow(chr_data_before_filter), "total genes)\n")
          cat("Debug: X-limits for", chr, ":", xlim[1], "to", xlim[2], "\n")
          cat("Debug: Genes outside have start positions:", range(outside_genes$start, na.rm = TRUE), "\n")
        }
      }
      
      if(nrow(chr_data) > 0) {
        # Debug: Print info for this chromosome
        cat("Debug: Chromosome", chr, "- plotting", nrow(chr_data), "measurement points\n")
        cat("Debug: Start range:", range(chr_data$start, na.rm = TRUE), "\n")
        cat("Debug: Y range:", range(chr_data$y_value_measurement, na.rm = TRUE), "\n")
        
        # Create alternating colors for chromosomes
        # Handle non-numeric chromosomes (like X, Y)
        if(grepl("^[0-9]+$", chr)) {
          chr_num <- as.numeric(chr)
          point_col <- if(chr_num %% 2 == 0) "#2E5943" else "#245780"  # Alternate dot colors
        } else {
          # Alternate colors for non-numeric chromosomes as well
          chr_idx <- which(c("X", "Y", "MT", "M") == chr)
          if(length(chr_idx) > 0) {
            point_col <- if(chr_idx %% 2 == 0) "#2E5943" else "#245780"
          } else {
            point_col <- "#245780"
          }
        }
        
        # Draw white grid lines for each tick (in background)
        for(i in seq(0, y_max_diseases, by = 25)) {
          circos.segments(0, i, max(xlim), i, col = "white", lwd = 0.5)
        }
        
        # Plot points for unique diseases
        circos.points(
          chr_data$start, 
          chr_data$y_value_diseases,
          col = point_col,
          pch = 16,
          cex = 0.5  # Smaller dot size
        )
        
        # Overlay highlighted points for labeled genes
        if(exists(".labeled_points") && nrow(.labeled_points) > 0) {
          lp_chr <- .labeled_points[.labeled_points$sector == chr, , drop = FALSE]
          if(nrow(lp_chr) > 0) {
            # Find y values for the labeled x positions within current chromosome data
            merge_df <- merge(lp_chr, chr_data[, c("start", "y_value_diseases")], by.x = "x", by.y = "start")
            if(nrow(merge_df) > 0) {
              circos.points(merge_df$x, merge_df$y_value_diseases, col = "#A01813", pch = 16, cex = 0.65)
            }
          }
        }
        
        # Add red circle line at level 9 for diseases
        # circos.segments(xlim[1], 9, xlim[2], 9,
        #                col = "white", lty = 1, lwd = 2)
        
        # Add y-axis for diseases (only on first chromosome)
        if(chr == "1") {
          circos.yaxis(side = "left", 
                      at = c(25, 50, 75, 100, 125, 150),
                      labels = c("25", "50", "75", "100", "125", "150"),
                      labels.cex = 0.55,
                      col = "#434343",
                      labels.col = "#434343")
        }
      }
    }
  )



  # Create the second Manhattan plot track (Unique Measurements)
  # y_max_measurement <- max(manhattan_data$y_value_measurement, na.rm = TRUE)
  # circos.track(
  #   ylim = c(0, y_max_measurement),
  #   bg.border = NA,
  #   bg.col = "lightgreen",  # Pale green background
  #   panel.fun = function(x, y) {
  #     chr = get.cell.meta.data("sector.index")
  #     xlim = get.cell.meta.data("xlim")
  #     ylim = get.cell.meta.data("ylim")
  #     
  #     # Filter data for current chromosome (show all data points)
  #     chr_data <- manhattan_data %>%
  #       filter(chromosome == chr) %>%
  #       filter(start >= xlim[1] & start <= xlim[2])
  #     
  #     if(nrow(chr_data) > 0) {
  #       # Debug: Print info for this chromosome
  #       cat("Debug: Chromosome", chr, "- plotting", nrow(chr_data), "therapeutic area points\n")
  #       cat("Debug: Y range:", range(chr_data$y_value_therapeutic, na.rm = TRUE), "\n")
  #       
  #       # Use same color scheme as diseases for consistency
  #       if(grepl("^[0-9]+$", chr)) {
  #         chr_num <- as.numeric(chr)
  #         point_col <- if(chr_num %% 2 == 0) "#2E8B57" else "#4682B4"
  #       } else {
  #         # For non-numeric chromosomes, use same colors as diseases
  #         chr_colors <- c("#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7")
  #         chr_idx <- which(c("X", "Y", "MT", "M") == chr)
  #         if(length(chr_idx) > 0) {
  #           point_col <- chr_colors[chr_idx]
  #         } else {
  #           point_col <- "#95A5A6"  # Default gray for other chromosomes
  #         }
  #       }
  #       
  #       # Plot points for unique measurements
  #       circos.points(
  #         chr_data$start, 
  #         chr_data$y_value_measurement,
  #         col = point_col,
  #         pch = 16,  # Same circle shape as diseases
  #         cex = 0.4  # Even smaller dot size
  #       )
  #       
  #       # Add green circle line at level 9 for measurements
  #       # circos.segments(xlim[1], 9, xlim[2], 9,
  #       #                col = "white", lty = 1, lwd = 2)
  #       
  #       # Add y-axis for measurements (only on first chromosome)
  #       if(chr == "1") {
  #         circos.yaxis(side = "left", 
  #                     at = c(0, 100, 200, 300, 400),
  #                     labels = c("0", "100", "200", "300", "400"),
  #                     labels.cex = 0.3,
  #                     tick.length = 0.1)
  #       }
  #     }
  #   }
  # )
  
  # Create the third Manhattan plot track (Unique Therapeutic Areas)
  y_max_therapeutic <- max(manhattan_data$y_value_therapeutic, na.rm = TRUE)
  circos.track(
    ylim = c(0, y_max_therapeutic),
    bg.border = NA,
    bg.col = "#A5CAE6",  # Therapeutic track background
    panel.fun = function(x, y) {
      chr = get.cell.meta.data("sector.index")
      xlim = get.cell.meta.data("xlim")
      ylim = get.cell.meta.data("ylim")
      
      # Filter data for current chromosome (show all data points)
      chr_data <- manhattan_data %>%
        filter(chromosome == chr) %>%
        filter(start >= xlim[1] & start <= xlim[2])
      
      if(nrow(chr_data) > 0) {
        # Debug: Print info for this chromosome
        cat("Debug: Chromosome", chr, "- plotting", nrow(chr_data), "disease points\n")
        cat("Debug: Y range:", range(chr_data$y_value_diseases, na.rm = TRUE), "\n")
        
        # Use same color scheme for consistency
        if(grepl("^[0-9]+$", chr)) {
          chr_num <- as.numeric(chr)
          point_col <- if(chr_num %% 2 == 0) "#2E5943" else "#245780"  # Alternate dot colors
        } else {
          # Alternate colors for non-numeric chromosomes as well
          chr_idx <- which(c("X", "Y", "MT", "M") == chr)
          if(length(chr_idx) > 0) {
            point_col <- if(chr_idx %% 2 == 0) "#2E5943" else "#245780"
          } else {
            point_col <- "#245780"
          }
        }
        
        # Draw white grid lines for each tick (in background)
        # Use smaller intervals for therapeutic areas since they have lower values
        for(i in seq(0, y_max_therapeutic, by = 5)) {
          circos.segments(0, i, max(xlim), i, col = "white", lwd = 0.5)
        }
        
        # Plot points for unique therapeutic areas
        circos.points(
          chr_data$start, 
          chr_data$y_value_therapeutic,
          col = point_col,
          pch = 16,  # Same circle shape as other tracks
          cex = 0.5  # Smaller dot size
        )
        
        # Overlay red highlights for genes with high disease counts on TA track
        # Use y_value_diseases since uniqueDiseases was renamed in prepare_manhattan_data
        if("y_value_diseases" %in% colnames(chr_data)) {
          high_dis_idx <- which(!is.na(chr_data$y_value_diseases) & chr_data$y_value_diseases > 35 &
                                 !is.na(chr_data$y_value_therapeutic))
          if(length(high_dis_idx) > 0) {
            circos.points(
              chr_data$start[high_dis_idx],
              chr_data$y_value_therapeutic[high_dis_idx],
              col = "#A01813",
              pch = 16,
              cex = 0.65
            )
          }
        }
        
        # Add blue circle line at level 5 for therapeutic areas
        # circos.segments(xlim[1], 5, xlim[2], 5,
        #                col = "white", lty = 1, lwd = 2)
        
        # Add y-axis for therapeutic areas (only on first chromosome)
        if(chr == "1") {
          circos.yaxis(side = "left", 
                      at = c(5, 10, 15, 20, 25),
                      labels = c("5", "10", "15", "20", "25"),
                      labels.cex = 0.55,
                      tick.length = 0.01,
                      col = "#434343",
                      labels.col = "#434343")
        }
      }
    }
  )

  # Add chromosome labels in the inner circle
  circos.track(
    ylim = c(0, 1),
    bg.col = NA,
    bg.border = NA,
    track.height = 0.00000000001,
    panel.fun = function(x, y) {
      chr = get.cell.meta.data("sector.index")
      xlim = get.cell.meta.data("xlim")
      ylim = get.cell.meta.data("ylim")
      circos.axis(h = "bottom", major.at = mean(xlim), labels = chr,
                  major.tick = FALSE, minor.ticks = 0,
                  labels.cex = 0.65, direction = "inside", labels.facing = "reverse.clockwise",
                  labels.niceFacing = TRUE, col = "white", labels.col = "#434343", labels.font = 2) 
    }
  )



  # Title removed as requested
  
  # # Add legend in the right corner below with better positioning
  # legend("bottomright", 
  #        legend = c("Diseases", "Measurements", "Therapeutic Areas"),
  #        fill = c("lightpink", "lightgreen", "lightblue"),
  #        title = "Data Tracks",
  #        cex = 0.7,
  #        bty = "n",
  #        xpd = TRUE,  # Allow plotting outside plot area
  #        inset = c(-0.02, 0.05))  # Move legend slightly outside



  
  # Add legend in the right corner below with better positioning
  legend("bottomright", 
         legend = c("Disease count", "TA count"),
         fill = c("#DBEAF6", "#8ABADE"),
         border = NA,
         title = "Data tracks",
         text.col = "#434343",
         cex = 0.9,
         text.font = 1,
         bty = "n",
         xpd = TRUE,  # Allow plotting outside plot area
         y.intersp = 1,
         inset = c(0.1, 0.07))  # Move legend slightly outside
         
  # # Draw a centered image inside the circle before clearing
  # img_path <- "/Users/polina/genetics_gsea/scr/8728108.png"
  # if(file.exists(img_path)) {
  #   img <- try(readPNG(img_path), silent = TRUE)
  #   if(!inherits(img, "try-error")) {
  #     # keep aspect ratio; fit within inner circle
  #     # canvas coordinates are roughly -1..1 both axes
  #     size <- 0.2
  #     rasterImage(img, -size, -size, size, size, interpolate = TRUE)
  #   }
  # }
  
  # Draw center plot in the inner hole (before circos.clear resets par)
  if (!is.null(center_plot)) {
    usr <- par("usr")   # user coord range, e.g. c(-1.07, 1.07, -1.07, 1.07)
    fig <- par("fig")   # figure region in device NDC: c(x1,x2,y1,y2)
    plt <- par("plt")   # plot region as fractions of figure: c(x1,x2,y1,y2)

    # Canvas centre (0,0) → device NDC
    px <- (0 - usr[1]) / (usr[2] - usr[1])          # 0..1 within plot region
    py <- (0 - usr[3]) / (usr[4] - usr[3])
    fx <- plt[1] + px * (plt[2] - plt[1])            # 0..1 within figure
    fy <- plt[3] + py * (plt[4] - plt[3])
    cx <- fig[1] + fx * (fig[2] - fig[1])            # device NDC x
    cy <- fig[3] + fy * (fig[4] - fig[3])            # device NDC y

    # center_r canvas units → NDC size (diameter of the viewport)
    r_ndc <- center_r / (usr[2] - usr[1]) * (plt[2] - plt[1]) * (fig[2] - fig[1])
    diam  <- r_ndc * 2

    grid::pushViewport(grid::viewport(
      x      = grid::unit(cx,   "npc"),
      y      = grid::unit(cy,   "npc"),
      width  = grid::unit(diam, "npc"),
      height = grid::unit(diam, "npc"),
      just   = c("centre", "centre")
    ))
    grid::grid.draw(ggplot2::ggplotGrob(center_plot))

    # Draw logo in the center hole of the donut (aspect-ratio preserved in inches)
    center_logo_path <- file.path(fig1_dir, "assets", "OT_helix_colour_RGB.png")
    if (file.exists(center_logo_path)) {
      logo_img <- png::readPNG(center_logo_path)
      img_w    <- dim(logo_img)[2]          # pixel width
      img_h    <- dim(logo_img)[1]          # pixel height
      img_ar   <- img_w / img_h             # true width-to-height ratio

      # Convert viewport size to physical inches so aspect ratio is exact
      dev_in    <- grDevices::dev.size("in")  # c(device_width_in, device_height_in)
      vp_w_in   <- diam * dev_in[1]           # physical width of the donut viewport
      vp_h_in   <- diam * dev_in[2]           # physical height of the donut viewport

      max_frac  <- 0.22                        # logo fills at most this fraction of the smaller dimension
      max_in    <- max_frac * min(vp_w_in, vp_h_in)

      if (img_ar >= 1) {                       # wider than tall → constrain width
        logo_w_in <- max_in
        logo_h_in <- max_in / img_ar
      } else {                                 # taller than wide → constrain height
        logo_h_in <- max_in
        logo_w_in <- max_in * img_ar
      }

      logo_grob <- grid::rasterGrob(logo_img, interpolate = TRUE,
        x      = grid::unit(0.5, "npc"),
        y      = grid::unit(0.5, "npc"),
        width  = grid::unit(logo_w_in, "inches"),
        height = grid::unit(logo_h_in, "inches"),
        just   = c("centre", "centre"))
      grid::grid.draw(logo_grob)
    }

    grid::popViewport()
  }

  # Clear the plot
  circos.clear()
  
  # Close device if output file specified (rendered directly)
  if(!is.null(output_file)) {
    dev.off()
    cat("Plot saved to:", output_file, "\n")
  }
}

# Main execution
main <- function() {
  # Path to the parquet file
  parquet_file <- file.path(fig1_dir, "data", "disease_ta_measur_index.snappy.parquet")
  
  cat("Loading data from:", parquet_file, "\n")
  
  # Read the data
  data <- read_parquet_data(parquet_file)
  
  cat("Data loaded successfully. Dimensions:", dim(data), "\n")
  cat("Column names:", colnames(data), "\n")
  
  # Display summary of the data
  cat("\nData summary:\n")
  print(summary(data))
  
  # Create the circular Manhattan plot
  cat("\nCreating circular Manhattan plot...\n")
  create_circular_manhattan(data, "circular_manhattan_plot_no_logo.png")
  
  cat("Circular Manhattan plot created successfully!\n")
}

# Run the main function
main()
