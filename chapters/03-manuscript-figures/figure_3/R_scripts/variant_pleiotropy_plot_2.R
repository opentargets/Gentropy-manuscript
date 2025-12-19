#!/usr/bin/env Rscript

##
## Variant pleiotropy plot from pre-exported CSV
##
## This script reproduces the scatter plot of
##  estimatedBeta vs -log10(p-value) coloured by therapeutic area,
## using the CSV produced by the Python script
##  `python_scripts/variant_pleiotropy_plot.py`.
##

suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(readr)
})

# --- Styling copied from Figure 2 (chapters/03-manuscript-figures/figure_2/figure_2.R) ---
# Theme to mimic matplotlib styling (Helvetica-like, light grid, no spines)
base_theme <- theme_minimal() +
    theme(
        text = element_text(face = "plain", color = "#434343"),
        plot.title = element_text(face = "plain", size = 10, hjust = 0.5, color = "#434343"),
        axis.title = element_text(size = 8, face = "plain", color = "#434343"),
        axis.text = element_text(size = 8, face = "plain", color = "#434343"),
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

# 5-color categorical palette (from Figure 2, plot C)
categorical_dark_colors <- c(
    "#A01813",
    "#D65A1F",
    "#30809e",
    "#d9d9d9"
)

# Small helper: use `y` when `x` is NULL/empty (avoids relying on rlang's `%||%`)
`%||%` <- function(x, y) {
    if (is.null(x) || length(x) == 0 || (is.character(x) && !nzchar(x[1]))) y else x
}

# Path to the exploded CSV (one row per study–therapeutic-area combination)
csv_path <- file.path(
    dirname(dirname(sys.frame(1)$ofile %||% "")),
    "data",
    "variant_pleiotropy_data_exploded_2.csv"
)

# Fallback if the above detection fails (e.g. in interactive use)
if (!file.exists(csv_path)) {
    csv_path <- "data/variant_pleiotropy_data_exploded_2.csv"
}

message("Reading data from: ", csv_path)

df <- readr::read_csv(csv_path, show_col_types = FALSE)

# Expected columns (from Python):
# - estimatedBeta
# - neg_log10_p
# - therapeuticAreaNames

df_plot <- df %>%
    mutate(
        therapeuticAreaNames = trimws(as.character(therapeuticAreaNames)),
        # Normalize any existing "other"/"Other"/"OTHER" category to a single label
        therapeuticAreaNames = if_else(tolower(therapeuticAreaNames) == "other", "Other", therapeuticAreaNames),
        diseaseNames = as.character(diseaseNames)
    )

# Count points per therapeutic area (before collapsing to "Other")
ta_counts <- df_plot %>%
    count(therapeuticAreaNames, name = "n_points") %>%
    arrange(desc(n_points))

message("\nPoints per therapeutic area (descending):")
print(ta_counts)

# Keep only the main 4 therapeutic areas (by number of points); merge the rest into "Other"
# (Never treat "Other" as a main TA)
n_main <- min(3, nrow(ta_counts))
top_tas <- ta_counts %>%
    filter(therapeuticAreaNames != "Other") %>%
    slice_head(n = n_main) %>%
    pull(therapeuticAreaNames)

# --- Legend order override ---
# Keep the "top N" behavior above, but force a preferred legend order for readability.
# Any TAs not listed here will follow afterwards (in their existing order), and "Other" stays last.
preferred_ta_order <- c(
    "nervous system disease",
    "cardiovascular disease"
)
top_tas <- c(intersect(preferred_ta_order, top_tas), setdiff(top_tas, preferred_ta_order))

# Helper function to capitalize first letter
capitalize_first <- function(x) {
    paste0(toupper(substring(x, 1, 1)), substring(x, 2))
}

# Capitalize therapeutic area names for legend
top_tas_capitalized <- capitalize_first(top_tas)
other_capitalized <- "Other"

df_plot <- df_plot %>%
    mutate(
        therapeuticAreaGroup = if_else(therapeuticAreaNames %in% top_tas, therapeuticAreaNames, "Other"),
        therapeuticAreaGroup = factor(therapeuticAreaGroup, levels = c(top_tas, "Other"))
    )

# 5 fixed colors: 4 main + "Other" (robust if there are <4 TAs in the input)
main_colors <- categorical_dark_colors[seq_along(top_tas)]
other_color <- categorical_dark_colors[min(length(categorical_dark_colors), length(top_tas) + 1)]
ta_colors <- c(setNames(main_colors, top_tas), Other = other_color)

# Create labels mapping for legend (capitalized)
ta_labels <- c(setNames(top_tas_capitalized, top_tas), Other = other_capitalized)

# Label specific diseases
label_df <- df_plot %>%
    filter(
        grepl("vascular dementia", diseaseNames, fixed = TRUE) |
            grepl("Alzheimer disease", diseaseNames, fixed = TRUE) |
            grepl("macular degeneration", diseaseNames, ignore.case = TRUE)
    ) %>%
    mutate(
        label = dplyr::case_when(
            grepl("vascular dementia", diseaseNames, fixed = TRUE) ~ "Vascular dementia",
            grepl("Alzheimer disease", diseaseNames, fixed = TRUE) ~ "Alzheimer disease",
            grepl("macular degeneration", diseaseNames, ignore.case = TRUE) ~ "Macular degeneration",
            TRUE ~ NA_character_
        )
    )

# One label per disease + many arrows to points with the same label
label_points <- label_df %>% filter(!is.na(label))

# Define position offsets for each label (adjustable per label)
label_positions <- data.frame(
    label = c("Vascular dementia", "Alzheimer disease", "Macular degeneration"),
    x_offset = c(-0.14, -0.1, -0.28), # Adjust x position (negative = left, positive = right)
    y_multiplier = c(0.5, 0.5, 0.5) # Adjust y position (multiplier for log scale)
)

label_anchors <- label_points %>%
    group_by(label) %>%
    summarise(
        max_x = max(estimatedBeta, na.rm = TRUE),
        max_y = max(neg_log10_p, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    left_join(label_positions, by = "label") %>%
    mutate(
        # Use custom offsets if defined, otherwise use defaults
        x_offset = if_else(is.na(x_offset), -0.08, x_offset),
        y_multiplier = if_else(is.na(y_multiplier), 0.5, y_multiplier),
        x = max_x + x_offset,
        y = max_y * y_multiplier
    ) %>%
    select(label, x, y)

label_segments <- label_points %>%
    inner_join(label_anchors, by = "label", suffix = c("_pt", "_lab")) %>%
    transmute(
        x = x,
        y = y,
        xend = estimatedBeta,
        yend = neg_log10_p,
        label = label
    )

p <- ggplot(df_plot, aes(
    x = estimatedBeta, y = neg_log10_p,
    colour = therapeuticAreaGroup
)) +
    geom_vline(xintercept = 0, colour = "#D65A1F", linetype = "dashed") +
    {
        # Draw many arrows to points, but only one label per disease
        if (nrow(label_points) > 0) {
            list(
                geom_segment(
                    data = label_segments,
                    aes(x = x, y = y, xend = xend, yend = yend),
                    inherit.aes = FALSE,
                    colour = "#bdbdbd",
                    linewidth = 0.3,
                    linetype = "dashed",
                    alpha = 0.45,
                    arrow = grid::arrow(length = grid::unit(0.15, "cm"))
                ),
                geom_label(
                    data = label_anchors,
                    aes(x = x, y = y, label = label),
                    inherit.aes = FALSE,
                    size = 3,
                    label.size = 0.2,
                    fill = "white",
                    colour = "#434343",
                    show.legend = FALSE
                )
            )
        } else {
            NULL
        }
    } +
    # Draw points after segments so arrows appear beneath the dots
    geom_point(size = 2, alpha = 0.85) +
    scale_colour_manual(
        values = ta_colors,
        breaks = levels(df_plot$therapeuticAreaGroup),
        labels = ta_labels,
        name = "Therapeutic Area"
    ) +
    scale_x_continuous(limits = c(-0.605, NA)) +
    scale_y_log10() +
    labs(
        title = "19_44908822_C_T",
        x = expression("Estimated " * beta),
        y = "-log10(p-value)"
    ) +
    base_theme +
    theme(
        # Keep legend in-plot (top-left) per your earlier request, but match Figure 2 text styling
        legend.position = c(0.02, 0.98),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = "white", colour = "#ececec"),
        legend.key = element_rect(fill = "white", colour = NA),
        legend.key.size = unit(0.5, "cm"),
        legend.key.width = unit(0.5, "cm"),
        legend.text = element_text(size = 8, color = "#434343"),
        legend.spacing.y = unit(0.01, "cm"),
        axis.text.x = element_text(size = 8, margin = margin(t = 2, b = 0), color = "#434343"),
        axis.title.x = element_text(size = 8, face = "plain", color = "#434343", margin = margin(t = 2))
    )

# print(p)

# Save plot to PNG file (always save, regardless of interactive/non-interactive mode)
out_path <- "/Users/polina/Gentropy-manuscript/chapters/03-manuscript-figures/figure_3/R_scripts/variant_pleiotropy_plot_R_2.png"
ggsave(out_path, p, width = 10, height = 3, dpi = 300)
message("Plot saved to: ", out_path)
