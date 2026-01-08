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
    # When running via source() or Rscript
    # Try to get script path from commandArgs (works with Rscript)
    script_path <- NULL
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
        script_path <- sub("^--file=", "", file_arg)
    } else {
        # Fallback: try sys.frame for source() calls
        tryCatch({
            script_path <- sys.frame(1)$ofile
        }, error = function(e) {
            script_path <- NULL
        })
    }
    if (!is.null(script_path)) {
        script_dir <- dirname(normalizePath(script_path))
        setwd(script_dir)
    }
    # If that doesn't work, script will use current working directory
}

# Theme to mimic matplotlib styling (Helvetica, light grid, no spines)
base_theme <- theme_minimal() +
    theme(
        text = element_text(face = "plain", color = "#434343"),
        plot.title = element_text(face = "plain", size = 10, hjust = 0.5, color = "#434343"),
        axis.title = element_text(size = 8, face = "plain", color = "#434343"),
        axis.text = element_text(size = 8, face = "plain", color = "#434343"),
        axis.text.x = element_text(size = 8, margin = margin(t = 2, b = 0), color = "#434343", angle = 45, hjust = 1),
        axis.title.x = element_text(size = 8, face = "plain", color = "#434343", margin = margin(t = 10)),
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
        legend.text = element_text(face = "plain", color = "#434343", size = 8)
    )

# Define colors (shared by plot a and plot b)
colors <- c(
    "cis-pqtl" = "#A01813",
    "eqtl" = "#E08145",
    "gwas-disease" = "#245780",
    "gwas-measurement" = "#2F735F"
)

# Ensure studyType has a consistent order across both plots
studytype_levels <- c("cis-pqtl", "eqtl", "gwas-disease", "gwas-measurement")

# ---- Plot A ----

# Read data for plot a)
maf_vep_dataset_part2 <- read.csv("figure_2_a.csv", stringsAsFactors = FALSE)
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
    theme(
        plot.margin = margin(t = 5, r = 5, b = 15, l = 5),
        axis.title.x = element_text(size = 8, face = "plain", color = "#434343", margin = margin(t = 10)),
        axis.title.y = element_text(size = 8, face = "plain", color = "#434343"),
        legend.position = "none" # Legend will be placed at bottom
    )

# ---- Plot B ----

# Read data for plot b)
maf_vep_dataset_all <- read.csv("figure_2_b.csv", stringsAsFactors = FALSE)

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
    theme(
        plot.margin = margin(t = 5, r = 5, b = 15, l = 5),
        axis.title.x = element_text(size = 8, face = "plain", color = "#434343", margin = margin(t = 10)),
        axis.title.y = element_text(size = 8, face = "plain", color = "#434343"),
        legend.position = "none" # Legend will be placed at bottom
    )

# ---- Plot C ----

# Categorical palette for consequence categories (used in plot c)
categorical_dark_colors <- c(
    "#BC3A19",
    "#E08145",
    "#E6CA9C",
    "#9EBAA8",
    "#2F735F"
)

# Expect data exported from Python as data2_for_plot_c.csv (data2 in figure_2.ipynb)
data_c <- read.csv("figure_2_c.csv", stringsAsFactors = FALSE)

# Set fillLabel factor order: smallest totalCountPerConsequence at bottom, largest at top
# ggplot2 stacks first factor level at bottom
fillLabel_order <- data_c %>%
    arrange(totalCountPerConsequence) %>%
    pull(fillLabel) %>%
    unique()

data_c$fillLabel <- factor(data_c$fillLabel, levels = fillLabel_order)

# Format labels: show rounded integers for segments > 5%
data_c <- data_c %>%
    mutate(
        label_formatted = ifelse(
            percentage > 0.05,
            sprintf("%.0f", round(percentage * 100)),
            ""
        )
    )

plot_c <- ggplot(
    data_c,
    aes(
        x = studyType,
        y = percentage,
        fill = fillLabel
    )
) +
    geom_col(width = 0.8, position = "stack") +
    geom_text(
        aes(label = label_formatted),
        position = position_stack(vjust = 0.5),
        colour = "white",
        size = 3
    ) +
    scale_y_continuous(
        labels = function(x) {
            # Remove 0 label, add % to all labels
            ifelse(x == 0, "", sprintf("%.0f%%", x * 100))
        }
    ) +
    scale_x_discrete(
        labels = c(
            "cis-pqtl" = "cis-pQTL",
            "eqtl" = "eQTL",
            "gwas-disease" = "GWAS (disease)",
            "gwas-measurement" = "GWAS (measurement)"
        ),
        expand = c(0, 0) # Reduce spacing between bars
    ) +
    scale_fill_manual(
        values = categorical_dark_colors,
        name = "Consequence category",
        labels = c(
            "protein_altering" = "Prot. altering",
            "promoter" = "Promoter",
            "intergenic" = "Intergenic",
            "enhancer" = "Enhancer",
            "intragenic" = "Intragenic"
        )
    ) +
    labs(
        x = "",
        y = ""
    ) +
    base_theme +
    theme(
        legend.position = "right",
        legend.text = element_text(size = 8, hjust = 0.5),
        legend.key.size = unit(0.45, "cm"),
        legend.key.height = unit(0.68, "cm"),
        legend.box.spacing = unit(0.05, "cm"),
        legend.spacing = unit(0.5, "cm"),
        axis.text.x = element_text(size = 8, margin = margin(t = 0, b = 0), color = "#434343", angle = 45, hjust = 0.95),
        axis.title.x = element_text(size = 8, face = "plain", color = "#434343", margin = margin(t = 10)),
        panel.spacing.x = unit(0, "cm"), # Reduce spacing between panels
        plot.margin = margin(t = 5, r = 0, b = 15, l = 0)
    )

# ---- Plot D ----

# Expect data exported from Python as combined_data_for_plot_d.csv (combined_data in figure_2.ipynb)
data_d <- read.csv("figure_2_d.csv", stringsAsFactors = FALSE)

# Use reverse order: protein_altering at top, intragenic at bottom
# For horizontal forest plot: first factor level appears at top of y-axis
data_d$consequence <- factor(data_d$consequence, levels = rev(fillLabel_order))

position_dodge_h <- position_dodge(width = 0.3)

plot_d <- ggplot(
    data_d,
    aes(
        x = meanAbsEstimatedBeta,
        y = consequence,
        color = study_category,
        group = study_category
    )
) +
    geom_errorbarh(
        aes(
            xmin = meanAbsEstimatedBeta - intervalAbsEstimatedBeta,
            xmax = meanAbsEstimatedBeta + intervalAbsEstimatedBeta
        ),
        height = 0.3,
        position = position_dodge_h
    ) +
    geom_point(
        position = position_dodge_h,
        size = 1.5
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#434343", linewidth = 0.5) +
    scale_color_manual(
        values = c("diseases" = "#245780", "measurements" = "#2F735F"),
        name = "Study type"
    ) +
    guides(
        color = guide_legend(
            nrow = 2, # 2 rows
            byrow = TRUE # Fill horizontally first
        )
    ) +
    labs(
        x = expression(mean("|" * hat(beta)[rescaled] * "|")),
        y = ""
    ) +
    base_theme +
    theme(
        legend.position = "bottom",
        legend.text = element_text(size = 8),
        legend.key.size = unit(0.5, "cm"),
        legend.direction = "horizontal",
        # legend.spacing.y = unit(0, "cm"),
        legend.box.spacing = unit(0.1, "cm"),
        legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
        axis.text.x = element_text(size = 8, color = "#434343", angle = 45, hjust = 1, margin = margin(t = 2, b = 0)),
        axis.text.y = element_blank(),
        axis.title.x = element_text(size = 8, face = "plain", color = "#434343", margin = margin(t = 10)),
        axis.title.y = element_blank(),
        plot.margin = margin(t = 5, r = 20, b = 15, l = 0)
    )

# ---- Combine all plots horizontally using cowplot ----

if (!requireNamespace("cowplot", quietly = TRUE)) {
    install.packages("cowplot")
}
library(cowplot)


# extract axis titles using get_plot_component (cowplot function)
axis_title_a <- get_plot_component(plot_a, "xlab-b", return_all = TRUE)
axis_title_b <- get_plot_component(plot_b, "xlab-b", return_all = TRUE)
axis_title_c <- get_plot_component(plot_c, "xlab-b", return_all = TRUE)
axis_title_d <- get_plot_component(plot_d, "xlab-b", return_all = TRUE)

rel_widths <- c(1, 1, 1.5, 1 / 2)

# Combine plots A and B without legends
plots_abcd <- plot_grid(
    plot_a + theme(legend.position = "none", axis.title.x = element_blank()),
    plot_b + theme(legend.position = "none", axis.title.x = element_blank()),
    plot_c + theme(legend.position = "right", axis.title.x = element_blank()),
    plot_d + theme(legend.position = "none", axis.title.x = element_blank()),
    nrow = 1,
    align = "h",
    rel_widths = rel_widths,
    labels = c("a", "b", "c", "d"),
    label_size = 8,
    label_x = c(0, 0, 0, -0.1)
)

# Combine plots A and B without legends
plots_abcd_x_axes <- plot_grid(
    axis_title_a,
    axis_title_b,
    axis_title_c,
    axis_title_d,
    nrow = 1,
    align = "h",
    rel_widths = rel_widths
)

# Extract legend from plot A (same legend for A and B)
legend_ab <- get_legend(plot_a + theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.text = element_text(size = 8),
    legend.key.size = unit(0.3, "cm"), # Adjust key size if needed
    legend.key.width = unit(0.5, "cm") # Adjust key width for horizontal legend
))

legend_c <- ggplot() +
    theme_void()

legend_d <- get_legend(plot_d + theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.text = element_text(size = 8), # Half of the default size 12
    legend.key.size = unit(0.3, "cm"), # Adjust key size if needed
    legend.key.width = unit(0.5, "cm") # Adjust key width for horizontal legend
))

rel_widths_ab_merged <- c(rel_widths[1] + rel_widths[2], rel_widths[3], rel_widths[4])

plot_legend_abcd <- plot_grid(
    legend_ab,
    legend_c,
    legend_d,
    nrow = 1,
    align = "h",
    rel_widths = rel_widths_ab_merged,
    label_size = 8
)


plots_abcd_x_axes_with_plots <- plot_grid(
    plots_abcd,
    plots_abcd_x_axes,
    nrow = 2,
    rel_heights = c(1, 0.2)
)

library(patchwork)

# Overlay plot_b on plot_a
plot_overlaid <- plots_abcd +
    inset_element(
        plots_abcd_x_axes,
        0, 0.8, 1, 0.3,
        align_to = "full"
    )

plot_overlaid_with_legend <- plot_overlaid +
    inset_element(
        plot_legend_abcd,
        0, 0, 1, 0.1,
        align_to = "full"
    )

ggsave("figure_2_new_colors.png", plot = plot_overlaid_with_legend, width = 8.27, height = 2.8, dpi = 300, bg = "#ffffff")
