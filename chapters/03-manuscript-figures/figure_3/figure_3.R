
suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(readr)
    library(tidyr)
})

text_colour <- "#434343"
grid_colour <- "#ececec"
axis_colour <- "#8a8a8a"
legend_text_size <- 8

# --- Styling copied from `figure_2/Figure_2.R` / `R_scripts/variant_pleiotropy_plot.R` ---
base_theme <- theme_minimal() +
    theme(
        text = element_text(face = "plain", color = text_colour, size = 8),
        axis.title = element_text(size = 8, face = "plain", color = text_colour),
        axis.text = element_text(size = 8, face = "plain", color = text_colour),
        axis.ticks = element_blank(),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = grid_colour, linewidth = 0.3),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_blank(),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(face = "plain", color = text_colour, size = legend_text_size),
        legend.key = element_rect(fill = "white", colour = NA),
        legend.background = element_rect(fill = "white", colour = NA),
        plot.title = element_text(size = 8, face = "plain", color = text_colour),
        plot.tag = element_text(face = "bold", size = 8, colour = text_colour)
    )

tag_theme <- function(x = 0) {
    theme(plot.tag.position = c(x, 1))
}

all_text_8_theme <- function() {
    theme(
        text = element_text(size = 8),
        plot.title = element_text(size = 8),
        axis.title = element_text(size = 8),
        axis.text = element_text(size = 8),
        legend.text = element_text(size = legend_text_size),
        legend.title = element_text(size = legend_text_size),
        plot.tag = element_text(size = 8, face = "bold")
    )
}

gap_theme <- function(pt = 5) {
    # Increase whitespace between panels when composing with patchwork/cowplot
    theme(plot.margin = margin(pt, pt, pt, pt, unit = "pt"))
}

legend_tight_theme <- function() {
    # Reduce whitespace between panel and bottom legend
    theme(
        legend.box.spacing = grid::unit(0, "pt"),
        legend.margin = margin(t = 4, r = 0, b = 0, l = 0, unit = "pt"),
        legend.box.margin = margin(t = 4, r = 0, b = 0, l = 0, unit = "pt")
    )
}

pal_model_type <- c(
    Univariate = "#245780",
    Joint = "#528B78"
)

# NOTE: user-updated palette (keep as-is)
pal_series <- c(
    "Observed (traits in cluster)" = "#3583C0",
    "Predicted (full model)" = "#A01813",
    "Predicted (no power)" = "#D65A1F"
)


get_script_path <- function() {
    # When called via source("..."), R sets `sys.frame(1)$ofile`
    ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
    if (!is.null(ofile) && nzchar(ofile)) {
        return(ofile)
    }

    # When called via Rscript, the script path is available as `--file=...`
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) == 1) {
        return(sub("^--file=", "", file_arg))
    }

    ""
}

# Helper function to capitalize first letter
capitalize_first <- function(x) {
    paste0(toupper(substring(x, 1, 1)), substring(x, 2))
}

# -----------------------------
# Pleiotropy plot generator function
# -----------------------------
create_pleiotropy_plot <- function(csv_path, plot_title, preferred_ta_order = character(0),
                                   palette = categorical_dark_colors, n_main_max = 3) {
    df <- readr::read_csv(csv_path, show_col_types = FALSE)

    df_plot <- df %>%
        mutate(
            therapeuticAreaNames = trimws(as.character(therapeuticAreaNames)),
            therapeuticAreaNames = if_else(tolower(therapeuticAreaNames) == "other", "Other", therapeuticAreaNames),
            diseaseNames = as.character(diseaseNames)
        )

    # Count points per therapeutic area (before collapsing to "Other")
    ta_counts <- df_plot %>%
        count(therapeuticAreaNames, name = "n_points") %>%
        arrange(desc(n_points))

    # Keep only the main N therapeutic areas (by number of points); merge the rest into "Other"
    n_main <- min(n_main_max, nrow(ta_counts))
    top_tas <- ta_counts %>%
        filter(therapeuticAreaNames != "Other") %>%
        slice_head(n = n_main) %>%
        pull(therapeuticAreaNames)

    # Optional legend-order override

    if (length(preferred_ta_order) > 0) {
        top_tas <- c(intersect(preferred_ta_order, top_tas), setdiff(top_tas, preferred_ta_order))
    }

    # Capitalize therapeutic area names for legend
    top_tas_capitalized <- capitalize_first(top_tas)
    other_capitalized <- "Other"

    df_plot <- df_plot %>%
        mutate(
            therapeuticAreaGroup = if_else(therapeuticAreaNames %in% top_tas, therapeuticAreaNames, "Other"),
            therapeuticAreaGroup = factor(therapeuticAreaGroup, levels = c(top_tas, "Other"))
        )

    # Count how many distinct therapeutic areas are collapsed into "Other"
    other_tas <- setdiff(unique(df$therapeuticAreaNames), c(top_tas, "Other"))
    other_n <- length(other_tas)

    # Colors: main + "Other"
    main_colors <- palette[seq_along(top_tas)]
    other_color <- palette[min(length(palette), length(top_tas) + 1)]
    ta_colors <- c(setNames(main_colors, top_tas), Other = other_color)

    # Create labels mapping for legend (capitalized, with Other count)
    ta_labels <- c(
        setNames(top_tas_capitalized, top_tas),
        Other = paste0("Other (", other_n, ")")
    )

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
        x_offset = c(-0.14, -0.1, -0.28),
        y_multiplier = c(0.5, 0.5, 0.5)
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
                        size = 2,
                        linewidth = 0,
                        fill = "white",
                        colour = text_colour,
                        show.legend = FALSE
                    )
                )
            } else {
                NULL
            }
        } +
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
            title = plot_title,
            x = expression("Estimated " * beta),
            y = "-log10(p-value)"
        ) +
        base_theme +
        theme(
            legend.position = c(0.02, 0.98),
            legend.justification = c(0, 1),
            legend.background = element_rect(fill = "white", colour = NA),
            legend.key = element_rect(fill = "white", colour = NA),
            legend.key.size = unit(0.5, "cm"),
            legend.key.width = unit(0.5, "cm"),
            legend.text = element_text(size = legend_text_size, color = text_colour),
            legend.spacing.y = unit(0.01, "cm"),
            axis.text.x = element_text(size = 8, margin = margin(t = 2, b = 0), color = text_colour),
            axis.title.x = element_text(size = 8, face = "plain", color = text_colour, margin = margin(t = 2))
        )

    p
}

# -----------------------------
# Resolve paths
# -----------------------------
script_path <- get_script_path()
figure_3_dir <- if (nzchar(script_path)) dirname(script_path) else ""

plot_a_path <- file.path(figure_3_dir, "data", "plot_a.csv")
plot_b_path <- file.path(figure_3_dir, "data", "plot_b.csv")
pleio_1_csv <- file.path(figure_3_dir, "data", "variant_pleiotropy_data_exploded.csv")
pleio_2_csv <- file.path(figure_3_dir, "data", "variant_pleiotropy_data_exploded_2.csv")

# Fallback for interactive use
if (!file.exists(plot_a_path)) plot_a_path <- "data/plot_a.csv"
if (!file.exists(plot_b_path)) plot_b_path <- "data/plot_b.csv"
if (!file.exists(pleio_1_csv)) pleio_1_csv <- "data/variant_pleiotropy_data_exploded.csv"
if (!file.exists(pleio_2_csv)) pleio_2_csv <- "data/variant_pleiotropy_data_exploded_2.csv"

if (!file.exists(plot_a_path)) stop("Could not find plot_a.csv at: ", plot_a_path)
if (!file.exists(plot_b_path)) stop("Could not find plot_b.csv at: ", plot_b_path)
if (!file.exists(pleio_1_csv)) stop("Could not find variant_pleiotropy_data_exploded.csv at: ", pleio_1_csv)
if (!file.exists(pleio_2_csv)) stop("Could not find variant_pleiotropy_data_exploded_2.csv at: ", pleio_2_csv)

message("Reading Plot a data from: ", plot_a_path)
message("Reading Plot b data from: ", plot_b_path)

df_a <- readr::read_csv(plot_a_path, show_col_types = FALSE)
df_b <- readr::read_csv(plot_b_path, show_col_types = FALSE)

is_plot_a_schema <- function(df) {
    required <- c("model_type", "coefficient", "ci_lower", "ci_upper", "y_numerical")
    all(required %in% names(df))
}

is_plot_b_schema <- function(df) {
    required <- c("maxMAF_bin")
    has_required <- all(required %in% names(df))
    # typical columns for plot b after python export
    has_some_series <- any(grepl("^(observed|predicted_.*)_(mean|sem)$", names(df)))
    has_required && has_some_series
}

# If the user renamed/swapped the input files, detect by schema and swap internally.
if (is_plot_b_schema(df_a) && is_plot_a_schema(df_b)) {
    message("Detected that plot_a_path contains Plot b schema and plot_b_path contains Plot a schema. Swapping inputs internally.")
    tmp <- df_a
    df_a <- df_b
    df_b <- tmp
} else if (!is_plot_a_schema(df_a)) {
    stop(
        "Plot a input does not match expected schema. Missing columns: ",
        paste(setdiff(c("model_type", "coefficient", "ci_lower", "ci_upper", "y_numerical"), names(df_a)), collapse = ", "),
        "\nInput file: ", plot_a_path
    )
} else if (!is_plot_b_schema(df_b)) {
    stop(
        "Plot b input does not match expected schema. Expected `maxMAF_bin` and *_mean/*_sem columns.\n",
        "Input file: ", plot_b_path
    )
}

# -----------------------------
# Plot a (forest-style)
# -----------------------------
df_a_plot <- df_a %>%
    mutate(
        # Be explicit about dplyr::recode() to avoid masking by other packages (e.g. car::recode)
        model_type = dplyr::recode(as.character(model_type), Multi = "Joint", .default = as.character(model_type)),
        model_type = factor(model_type, levels = c("Univariate", "Joint"))
    )

y_breaks <- df_a_plot %>%
    distinct(y_numerical, covariate_label) %>%
    arrange(y_numerical)

p_a <- ggplot(df_a_plot, aes(
    x = coefficient,
    y = y_plot,
    colour = model_type
)) +
    geom_vline(xintercept = 0, colour = "#bdbdbd", linetype = "dashed", linewidth = 0.4) +
    # ggplot2 >= 4.0 deprecated geom_errorbarh(); use geom_errorbar() + orientation instead
    geom_errorbar(
        aes(xmin = ci_lower, xmax = ci_upper),
        orientation = "y",
        width = 0,
        linewidth = 0.4,
        alpha = 0.9
    ) +
    geom_point(size = 1.8, alpha = 0.95) +
    scale_colour_manual(values = pal_model_type, name = NULL) +
    scale_y_continuous(
        breaks = y_breaks$y_numerical,
        labels = y_breaks$covariate_label
    ) +
    labs(
        tag = "b",
        x = "Coefficient (95% CI)",
        y = NULL
    ) +
    base_theme +
    all_text_8_theme() +
    tag_theme(x = -0.03) + # Nudge tag left a bit for the right-hand panel
    gap_theme(12) +
    legend_tight_theme() +
    theme(
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = grid_colour, linewidth = 0.3),
        axis.line = element_line(color = axis_colour, linewidth = 0.3),
        legend.position = "right"
    )

# -----------------------------
# Plot b (binned observed vs predicted, 95% CI)
# -----------------------------
maf_levels <- c("0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5")

df_b_long <- df_b %>%
    mutate(maxMAF_bin = factor(as.character(maxMAF_bin), levels = maf_levels)) %>%
    pivot_longer(
        cols = -maxMAF_bin,
        names_to = c("series", "stat"),
        names_pattern = "^(.*)_(mean|sem)$",
        values_to = "value"
    ) %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(
        ci = sem * 1.96,
        series = dplyr::recode(series,
            observed = "Observed (traits in cluster)",
            predicted_full = "Predicted (full model)",
            predicted_no_power = "Predicted (no power)"
        ),
        series = factor(series, levels = c(
            "Observed (traits in cluster)",
            "Predicted (full model)",
            "Predicted (no power)"
        ))
    )

p_b <- ggplot(df_b_long, aes(
    x = maxMAF_bin,
    y = mean,
    colour = series,
    fill = series,
    group = series
)) +
    # CI as semi-transparent area (Figure_2 style)
    geom_ribbon(aes(ymin = mean - ci, ymax = mean + ci), alpha = 0.12, linewidth = 0, na.rm = TRUE) +
    geom_line(linewidth = 0.6) +
    scale_colour_manual(values = pal_series, name = NULL) +
    # Let fill+colour merge into a single legend so the key shows the translucent CI band behind the line (Figure 2 style)
    scale_fill_manual(values = pal_series, name = NULL) +
    # Remove default padding so panel (and horizontal gridlines) ends at first/last x bin
    scale_x_discrete(expand = c(0, 0)) +
    labs(
        tag = "a",
        x = "MAF bin",
        y = "Number of traits"
    ) +
    base_theme +
    all_text_8_theme() +
    tag_theme(x = 0) +
    gap_theme(12) +
    legend_tight_theme() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.line = element_line(color = axis_colour, linewidth = 0.3),
        axis.title.y = element_text(margin = margin(r = -4)),
        legend.position = "right",
        legend.text = element_text(size = legend_text_size),
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.3, "cm"),
        legend.box.margin = margin(l = 8)
    )

# -----------------------------
# Plot c and d (variant pleiotropy plots with shared disease-based legend)
# -----------------------------
message("Reading pleiotropy data from: ", pleio_1_csv)
message("Reading pleiotropy data from: ", pleio_2_csv)

# Read both datasets
df_pleio_1 <- readr::read_csv(pleio_1_csv, show_col_types = FALSE)
df_pleio_2 <- readr::read_csv(pleio_2_csv, show_col_types = FALSE)

# Combine both datasets to find top 5 diseases by point count
df_combined <- bind_rows(
    df_pleio_1 %>% mutate(plot = "c"),
    df_pleio_2 %>% mutate(plot = "d")
) %>%
    mutate(
        diseaseNames = trimws(as.character(diseaseNames))
    )

# Count points per disease across both plots
disease_counts <- df_combined %>%
    count(diseaseNames, name = "n_points") %>%
    arrange(desc(n_points))

# Keep only the top 5 diseases by number of points
top_diseases <- disease_counts %>%
    slice_head(n = 5) %>%
    pull(diseaseNames)

# Capitalize disease names for legend
top_diseases_capitalized <- capitalize_first(top_diseases)

# Create a shared color scheme for both plots (top 5 diseases + Other)
disease_palette <- c("#A01813", "#E3A772", "#2E5943", "#4F97CF", "#D65A1F")
disease_colors <- c(
    setNames(disease_palette[1:length(top_diseases)], top_diseases),
    Other = "#d9d9d9"
)

# Create labels for legend (capitalized)
disease_labels <- c(
    setNames(top_diseases_capitalized, top_diseases),
    Other = "Other"
)

# Function to prepare disease-based plot data
prepare_disease_plot <- function(df, top_diseases) {
    df_plot <- df %>%
        mutate(
            diseaseNames = trimws(as.character(diseaseNames)),
            diseaseGroup = if_else(diseaseNames %in% top_diseases, diseaseNames, "Other"),
            diseaseGroup = factor(diseaseGroup, levels = c(top_diseases, "Other"))
        )
    
    df_plot
}

# Prepare data for both plots
plot_c_data <- prepare_disease_plot(df_pleio_1, top_diseases)
plot_d_data <- prepare_disease_plot(df_pleio_2, top_diseases)

# Create plot c
p_c1 <- ggplot(plot_c_data, aes(
    x = estimatedBeta, y = neg_log10_p,
    colour = diseaseGroup
)) +
    geom_vline(xintercept = 0, colour = "#D65A1F", linetype = "dashed") +
    geom_point(size = 2, alpha = 0.85) +
    scale_colour_manual(
        values = disease_colors,
        breaks = levels(plot_c_data$diseaseGroup),
        labels = disease_labels,
        name = NULL
    ) +
    scale_x_continuous(limits = c(-0.605, NA)) +
    scale_y_log10() +
    labs(
        tag = "c",
        x = expression("Estimated " * beta * " for 19_44908684_T_C"),
        y = "-log10(p-value)"
    ) +
    base_theme +
    all_text_8_theme() +
    gap_theme(5) +
    theme(
        legend.position = "none",  # Remove legend from plot c
        plot.tag = element_text(face = "bold", size = 8, colour = text_colour),
        plot.tag.position = c(0, 1),
        axis.line = element_line(color = axis_colour, linewidth = 0.3)
    )

# Create plot d
p_c2 <- ggplot(plot_d_data, aes(
    x = estimatedBeta, y = neg_log10_p,
    colour = diseaseGroup
)) +
    geom_vline(xintercept = 0, colour = "#D65A1F", linetype = "dashed") +
    geom_point(size = 2, alpha = 0.85) +
    scale_colour_manual(
        values = disease_colors,
        breaks = levels(plot_d_data$diseaseGroup),
        labels = disease_labels,
        name = NULL
    ) +
    scale_x_continuous(limits = c(-0.605, NA)) +
    scale_y_log10() +
    labs(
        tag = "d",
        x = expression("Estimated " * beta * " for 19_44908822_C_T"),
        y = "-log10(p-value)"
    ) +
    base_theme +
    all_text_8_theme() +
    gap_theme(12) +
    theme(
        legend.position = c(0.98, 0.02),  # Bottom right corner
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.box.background = element_rect(fill = "white", colour = NA),
        legend.key = element_rect(fill = "white", colour = NA),
        legend.key.size = unit(0.5, "cm"),
        legend.key.width = unit(0.5, "cm"),
        legend.text = element_text(size = legend_text_size, color = text_colour),
        legend.spacing.y = unit(0.01, "cm"),
        plot.tag = element_text(face = "bold", size = 8, colour = text_colour),
        plot.tag.position = c(0, 1),
        axis.line = element_line(color = axis_colour, linewidth = 0.3)
    )

# Align x-axis limits for plots c and d to ensure zero coordinates match
x_range_c <- ggplot_build(p_c1)$layout$panel_params[[1]]$x.range
x_range_d <- ggplot_build(p_c2)$layout$panel_params[[1]]$x.range
x_min <- min(x_range_c[1], x_range_d[1])
x_max <- max(x_range_c[2], x_range_d[2])

p_c1 <- p_c1 + scale_x_continuous(limits = c(x_min, x_max))
p_c2 <- p_c2 + scale_x_continuous(limits = c(x_min, x_max))

# -----------------------------
# Combine horizontally and save
# -----------------------------
out_path <- file.path(figure_3_dir, "figure_3_reverted.png")
if (!nzchar(figure_3_dir)) out_path <- "figure_3_reverted.png"

combined <- NULL
if (requireNamespace("patchwork", quietly = TRUE)) {
    suppressPackageStartupMessages(library(patchwork))
    # Layout:
    # Row 1: a | c
    # Row 2: b | d
    # Width ratio: left column (a,b) : right column (c,d) = 1:3
    design <- "
AC
BD
"
    combined <- p_b + p_a + p_c1 + p_c2 +
        plot_layout(design = design, widths = c(1, 3))
} else if (requireNamespace("cowplot", quietly = TRUE)) {
    suppressPackageStartupMessages(library(cowplot))
    top_row <- cowplot::plot_grid(p_b, p_c1, ncol = 2, rel_widths = c(1, 3))
    bottom_row <- cowplot::plot_grid(p_a, p_c2, ncol = 2, rel_widths = c(1, 3))
    combined <- cowplot::plot_grid(top_row, bottom_row, ncol = 1, rel_heights = c(1, 1))
} else {
    stop("Please install either 'patchwork' or 'cowplot' to combine the two plots into a grid.")
}

# Save at A4 portrait width (inches). Keep previous aspect ratio (12x4) to avoid distortion.
a4_portrait_width_in <- 8.27
aspect_ratio <- 2 / 12
ggsave(out_path, combined, width = a4_portrait_width_in, height = a4_portrait_width_in * aspect_ratio * 3, units = "in", dpi = 300, bg = "white")
message("Saved: ", out_path)
