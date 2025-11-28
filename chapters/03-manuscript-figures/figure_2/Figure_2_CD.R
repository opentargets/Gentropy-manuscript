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

# Categorical palette for consequence categories (used in plot c)
categorical_dark_colors <- c(
    # "#fa4d56",
    # "#002d9c",
    # "#009d9a",
    # "#a56eff",
    # "#005d5d",
    # "#DBEAF6",
    # "#BFDAEE",
    # "#A5CAE6",
    # "#8ABADE",
    # "#6EA9D7",
    # "#4F97CF",
    # "#3583C0",
    # "#A5CAE6",
    # "#e1b400",
    # "#FF6350",
    # "#3489CA",
    # # "#2C6EA0",
    # "#2F735F",
    # "#E08145",
    # "#245780"
    # A01813,
    #   "#A01813",
    "#BC3A19",
    #   "#D65A1F",
    "#E08145",
    #   "#E3A772",
    "#E6CA9C",
    #   "#ECEADA",
    #   "#C5D2C1",
    "#9EBAA8",
    #   "#78A290",
    #   "#528B78",
    "#2F735F"
    #   "#2E5943",
)

# ---- Plot c: barplot of consequence proportions per study type (from Python data2) ----

# Expect data exported from Python as data2_for_plot_c.csv (data2 in figure_2.ipynb)
data_c <- read.csv("data2_for_plot_c.csv", stringsAsFactors = FALSE)

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
    geom_col(width = 0.7, position = "stack") +
    geom_text(
        aes(label = label_formatted),
        position = position_stack(vjust = 0.5),
        colour = "white",
        size = 3
    ) +
    scale_y_continuous(
        labels = function(x) {
            # Remove 0 label and remove % from all labels
            ifelse(x == 0, "", sprintf("%.0f", x * 100))
        }
    ) +
    scale_x_discrete(
        labels = c(
            "cis-pqtl" = "cis-pQTL",
            "eqtl" = "eQTL",
            "gwas-disease" = "GWAS (disease)",
            "gwas-measurement" = "GWAS (measurement)"
        )
    ) +
    scale_fill_manual(values = categorical_dark_colors, name = "Consequence category") +
    labs(
        x = "",
        y = "%"
    ) +
    base_theme +
    theme(
        legend.position = "right",
        plot.margin = margin(t = 5, r = 5, b = 15, l = 5)
    )

# ---- Plot d: forest plot of mean |beta_rescaled| per consequence (from Python combined_data) ----

# Expect data exported from Python as combined_data_for_plot_d.csv (combined_data in figure_2.ipynb)
data_d <- read.csv("combined_data_for_plot_d.csv", stringsAsFactors = FALSE)

# Ensure ordering of consequence based on diseases meanAbsEstimatedBeta (highest to lowest)
consequence_order <- data_d %>%
    dplyr::filter(study_category == "diseases") %>%
    dplyr::arrange(dplyr::desc(meanAbsEstimatedBeta)) %>%
    dplyr::pull(consequence)

data_d$consequence <- factor(data_d$consequence, levels = consequence_order)

position_dodge_w <- position_dodge(width = 0.3)

plot_d <- ggplot(
    data_d,
    aes(
        x = consequence,
        y = meanAbsEstimatedBeta,
        color = study_category,
        group = study_category
    )
) +
    geom_errorbar(
        aes(
            ymin = meanAbsEstimatedBeta - intervalAbsEstimatedBeta,
            ymax = meanAbsEstimatedBeta + intervalAbsEstimatedBeta
        ),
        width = 0.3,
        position = position_dodge_w
    ) +
    geom_point(
        position = position_dodge_w,
        size = 1.5
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.5) +
    scale_color_manual(
        values = c("diseases" = "#245780", "measurements" = "#2F735F"),
        name = "Study type"
    ) +
    labs(
        x = "",
        y = expression(mean("|" * hat(beta)[rescaled] * "|"))
    ) +
    base_theme +
    coord_flip() +
    theme(
        legend.position = "right",
        plot.margin = margin(t = 5, r = 5, b = 15, l = 5)
    )

# ---- Combine plots C and D side by side ----

# Build grobs for each panel
g_c <- ggplotGrob(plot_c)
g_d <- ggplotGrob(plot_d)

# Align heights so that axes and panels line up
max_height <- grid::unit.pmax(g_c$heights, g_d$heights)
g_c$heights <- max_height
g_d$heights <- max_height

# Bind columns side by side
cbind_g <- getFromNamespace("cbind_gtable", "gtable")
combined_grob <- cbind_g(g_c, g_d, size = "max")

# Save combined plot as png (plots C and D side by side)
ggsave("figure_2_CD.png", plot = combined_grob, width = 8.27, height = 3, dpi = 300, bg = "#ffffff00")

# Return the combined plot object (can be used in faceting or printed)
combined_grob
