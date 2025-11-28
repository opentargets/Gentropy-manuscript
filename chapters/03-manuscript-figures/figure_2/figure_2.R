suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
})

# Set working directory to the script's location
# This ensures the script runs from its own directory, not the root
if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    # When running in RStudio
    script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
    setwd(script_dir)
} else {
    # When running via source() or command line
    # Try to get script path from source() call
    script_path <- sys.frame(1)$ofile
    if (!is.null(script_path)) {
        script_dir <- dirname(normalizePath(script_path))
        setwd(script_dir)
    }
    # If that doesn't work, script will use current working directory
}

# Read data for plot a)
maf_vep_dataset_part2 <- read.csv("for_figure_2_part_2.csv", stringsAsFactors = FALSE)

# Ensure studyType has a consistent order across both plots
studytype_levels <- c("cis-pqtl", "eqtl", "gwas-disease", "gwas-measurement")
maf_vep_dataset_part2$studyType <- factor(maf_vep_dataset_part2$studyType, levels = studytype_levels)

# X-axis breaks for plot a) (same as in the python version)
x_breaks_a <- sort(unique(maf_vep_dataset_part2$midPoint))

# Calculate y-axis limits for plot a) to ensure all data is visible
max_y_a <- max(
    maf_vep_dataset_part2$meanAbsEstimatedBeta + maf_vep_dataset_part2$intervalAbsEstimatedBeta,
    na.rm = TRUE
)
y_upper_a <- ceiling(max_y_a * 10) / 10 # Round up to nearest 0.1

# Create plot a)
plot_a <- ggplot(
    maf_vep_dataset_part2,
    aes(
        x = midPoint,
        y = meanAbsEstimatedBeta,
        color = studyType,
        fill = studyType,
        group = studyType
    )
) +
    geom_ribbon(
        aes(
            ymin = meanAbsEstimatedBeta - intervalAbsEstimatedBeta,
            ymax = meanAbsEstimatedBeta + intervalAbsEstimatedBeta
        ),
        alpha = 0.12,
        linewidth = 0,
        na.rm = TRUE
    ) +
    geom_line(linewidth = 0.5, na.rm = TRUE) +
    scale_color_manual(
        values = colors,
        breaks = names(colors),
        labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
        name = "studyType"
    ) +
    scale_fill_manual(
        values = colors,
        breaks = names(colors),
        labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
        name = "studyType"
    ) +
    scale_x_continuous(
        breaks = x_breaks_a,
        labels = x_breaks_a,
        expand = c(0, 0)
    ) +
    labs(
        x = expression(mean(MAF) %+-% 0.025),
        y = expression(mean("|" * hat(beta)[rescaled] * "|"))
    ) +
    base_theme +
    coord_cartesian(ylim = c(0, y_upper_a)) +
    theme(plot.margin = margin(t = 5, r = 5, b = 15, l = 5))

# Read data for plot b)
maf_vep_dataset_all <- read.csv("for_figure_2_part_1.csv", stringsAsFactors = FALSE)

# Ensure studyType has the same factor levels as plot a)
maf_vep_dataset_all$studyType <- factor(maf_vep_dataset_all$studyType, levels = studytype_levels)

# Create copy and convert PAV to integer
df <- maf_vep_dataset_all
# PAV column contains "True"/"False" strings, so convert to logical first, then integer
df$PAV_int <- as.integer(df$PAV == "True")

# Define bins and labels
bins <- c(0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
labels <- c("<0.01", "0.01–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5")

# Create MAF bins
df$maf_bin <- cut(df$value, breaks = bins, labels = labels, include.lowest = TRUE)

# Group by studyType + maf_bin and calculate statistics
bin_stats <- df %>%
    group_by(studyType, maf_bin) %>%
    summarise(
        p = mean(PAV_int, na.rm = TRUE),
        n = n(),
        n_pav = sum(PAV_int, na.rm = TRUE),
        .groups = "drop"
    )

# Calculate SE and confidence intervals
bin_stats <- bin_stats %>%
    mutate(
        SE = sqrt(p * (1 - p) / n),
        CI_lower = pmax(0, pmin(1, p - 1.96 * SE)),
        CI_upper = pmax(0, pmin(1, p + 1.96 * SE))
    )

# Theme to mimic matplotlib styling (Helvetica, light grid, no spines)
base_theme <- theme_minimal() +
    theme(
        text = element_text(face = "plain", color = "#434343"),
        plot.title = element_text(face = "plain", size = 10, hjust = 0.5, color = "#434343"),
        axis.title = element_text(size = 12, face = "plain", color = "#434343"),
        axis.text = element_text(size = 10, face = "plain", color = "#434343"),
        axis.text.x = element_text(size = 10, margin = margin(t = 2, b = 0), color = "#434343", angle = 45, hjust = 1),
        axis.title.x = element_text(size = 12, face = "plain", color = "#434343", margin = margin(t = 10)),
        axis.ticks = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "#ececec", linewidth = 0.3),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank(),
        legend.position = "right",
        legend.title = element_blank(),
        legend.text = element_text(face = "plain", color = "#434343", size = 12)
    )

# Define colors (shared by plot a and plot b)
colors <- c(
    "cis-pqtl" = "#A01813",
    "eqtl" = "#E08145",
    "gwas-disease" = "#245780",
    "gwas-measurement" = "#2F735F"
)

# Calculate y-axis limits to ensure all data is visible in plot b)
max_y <- max(bin_stats$CI_upper, na.rm = TRUE)
y_upper <- ceiling(max_y * 10) / 10 # Round up to nearest 0.1

# Create plot b)
plot_b <- ggplot(bin_stats, aes(x = maf_bin, y = p, color = studyType, fill = studyType, group = studyType)) +
    geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper), alpha = 0.12, linewidth = 0, na.rm = TRUE) +
    geom_line(linewidth = 0.5, na.rm = TRUE) +
    scale_color_manual(
        values = colors,
        breaks = names(colors),
        labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
        name = "studyType"
    ) +
    scale_fill_manual(
        values = colors,
        breaks = names(colors),
        labels = c("cis-pQTL", "eQTL", "GWAS (disease)", "GWAS (measurement)"),
        name = "studyType"
    ) +
    scale_x_discrete(expand = c(0, 0)) +
    labs(
        x = "MAF bins",
        y = "Proportion of PAV"
    ) +
    base_theme +
    coord_cartesian(ylim = c(0, max(y_upper, 0.6))) +
    theme(plot.margin = margin(t = 5, r = 5, b = 15, l = 5))

# Combine plot a) (left, without legend) and plot b) (right, with legend on the right) into a single grid
g_a <- ggplotGrob(plot_a + theme(legend.position = "none"))
g_b <- ggplotGrob(plot_b)

# Align heights so that axes and panels line up
max_height <- grid::unit.pmax(g_a$heights, g_b$heights)
g_a$heights <- max_height
g_b$heights <- max_height

cbind_g <- getFromNamespace("cbind_gtable", "gtable")
combined_grob <- cbind_g(g_a, g_b, size = "max")

# Save combined plot as png (A4 landscape: 297mm x 210mm ≈ 11.69 x 8.27 inches)
ggsave("figure_2.png", plot = combined_grob, width = 8.27, height = 3, dpi = 300, bg = "#ffffff00")

# Return the combined plot object (can be used in faceting or printed)
combined_grob
