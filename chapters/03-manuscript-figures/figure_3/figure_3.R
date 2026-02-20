
suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(readr)
    library(tidyr)
})

text_colour <- "#434343"
grid_colour <- "#ececec"
axis_colour <- "#8a8a8a"
legend_text_size <- 7

# --- Styling copied from `figure_2/Figure_2.R` / `R_scripts/variant_pleiotropy_plot.R` ---
base_theme <- theme_minimal() +
    theme(
        text = element_text(face = "plain", color = text_colour, size = 7),
        axis.title = element_text(size = 7, face = "plain", color = text_colour),
        axis.text = element_text(size = 7, face = "plain", color = text_colour),
        axis.ticks = element_line(color = axis_colour, linewidth = 0.3),
        axis.ticks.length = unit(2, "pt"),
        panel.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(color = axis_colour, linewidth = 0.3),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(face = "plain", color = text_colour, size = legend_text_size),
        legend.key = element_rect(fill = "white", colour = NA),
        legend.background = element_rect(fill = "white", colour = NA),
        plot.title = element_text(size = 7, face = "plain", color = text_colour),
        plot.tag = element_text(face = "bold", size = 7, colour = text_colour)
    )

tag_theme <- function(x = 0) {
    theme(plot.tag.position = c(x, 1))
}

all_text_8_theme <- function() {
    theme(
        text = element_text(size = 7),
        plot.title = element_text(size = 7),
        axis.title = element_text(size = 7),
        axis.text = element_text(size = 7),
        legend.text = element_text(size = legend_text_size),
        legend.title = element_text(size = legend_text_size),
        plot.tag = element_text(size = 7, face = "bold")
    )
}

gap_theme <- function(pt = 5) {
    theme(plot.margin = margin(pt, pt, pt, pt, unit = "pt"))
}

legend_tight_theme <- function() {
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
    "Observed" = "#3583C0",
    "Predicted (full)" = "#A01813",
    "Predicted (np)" = "#D65A1F" 
)


get_script_path <- function() {
    ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
    if (!is.null(ofile) && nzchar(ofile)) {
        return(ofile)
    }

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
    geom_errorbar(
        aes(xmin = ci_lower, xmax = ci_upper),
        orientation = "y",
        width = 0,
        linewidth = 0.3,
        alpha = 0.9
    ) +
    geom_point(size = 1, alpha = 1) +
    scale_colour_manual(values = pal_model_type, name = NULL) +
    scale_x_continuous(breaks = c(0.5, 1.0, 1.5, 2.0), limits = c(NA, 2.0), expand = expansion(mult = c(0.05, 0))) +
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
    theme(plot.tag.position = "topleft") +
    gap_theme(2) +
    legend_tight_theme() +
    theme(
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        legend.position = "right",
        legend.spacing.y = unit(0, "pt"),
        legend.key.height = unit(0.3, "cm"),
        axis.title.x = element_text(margin = margin(t = 8)),
        aspect.ratio = 1
    )

# -----------------------------
# Plot b (binned observed vs predicted, 95% CI)
# -----------------------------
maf_levels <- c("0-0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5")

# Extract lower bound of each MAF bin for continuous positioning
maf_lower <- as.numeric(sub("-.*", "", maf_levels))
maf_bin_to_x <- setNames(maf_lower, maf_levels)
# Remove label (but keep tick) for second interval (0.01-0.05)
maf_labels <- ifelse(maf_levels == "0.01-0.05", "", maf_levels)

df_b_long <- df_b %>%
    mutate(maxMAF_bin = as.character(maxMAF_bin)) %>%
    pivot_longer(
        cols = -maxMAF_bin,
        names_to = c("series", "stat"),
        names_pattern = "^(.*)_(mean|sem)$",
        values_to = "value"
    ) %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(
        ci = sem * 1.96,
        maf_x = maf_bin_to_x[maxMAF_bin],
        series = dplyr::recode(series,
            observed = "Observed",
            predicted_full = "Predicted (full)",
            predicted_no_power = "Predicted (np)"
        ),
        series = factor(series, levels = c(
            "Observed",
            "Predicted (full)",
            "Predicted (np)"
        ))
    )

p_b <- ggplot(df_b_long, aes(
    x = maf_x,
    y = mean,
    colour = series,
    fill = series,
    group = series
)) +
    geom_ribbon(aes(ymin = mean - ci, ymax = mean + ci), alpha = 0.12, linewidth = 0, na.rm = TRUE) +
    geom_line(linewidth = 0.3) +
    scale_colour_manual(values = pal_series, name = NULL) +
    scale_fill_manual(values = pal_series, name = NULL) +
    scale_x_continuous(breaks = maf_lower, labels = maf_labels, limits = c(0, 0.4), expand = c(0, 0)) +
    labs(
        tag = "a",
        x = "MAF bin",
        y = "Number of traits"
    ) +
    base_theme +
    all_text_8_theme() +
    gap_theme(12) +
    legend_tight_theme() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.y = element_text(margin = margin(t = 0, r = 6, b = 0, l = 0)),
        plot.margin = margin(t = 0, r = 0, b = 0, l = 0),
        plot.tag.position = "topleft",
        legend.position = "right",
        legend.text = element_text(size = legend_text_size),
        legend.key.size = unit(0.3, "cm"),
        legend.key.width = unit(0.3, "cm"),
        legend.box.margin = margin(l = 8),
        aspect.ratio = 1
    )

# -----------------------------
# Panel c: faceted variant pleiotropy scatter plot
# -----------------------------
message("Reading pleiotropy data from: ", pleio_1_csv)
message("Reading pleiotropy data from: ", pleio_2_csv)

df_pleio_1 <- readr::read_csv(pleio_1_csv, show_col_types = FALSE)
df_pleio_2 <- readr::read_csv(pleio_2_csv, show_col_types = FALSE)

# Combine both datasets with variant labels as faceting variable
df_combined <- bind_rows(
    df_pleio_1 %>% mutate(variant = "APOE-Cys130"),
    df_pleio_2 %>% mutate(variant = "APOE-Arg176")
) %>%
    mutate(
        diseaseNames = trimws(as.character(diseaseNames)),
        diseaseNames = gsub("^\\['|'\\]$", "", diseaseNames),
        diseaseNames = gsub("^\\[|\\]$", "", diseaseNames),
        diseaseNames = trimws(diseaseNames),
        variant = factor(variant, levels = c("APOE-Cys130", "APOE-Arg176"))
    )

# Count points per disease across both variants
disease_counts <- df_combined %>%
    count(diseaseNames, name = "n_points") %>%
    arrange(desc(n_points))

# Keep top 7 diseases by number of points
top_diseases <- disease_counts %>%
    slice_head(n = 7) %>%
    pull(diseaseNames)

# Capitalize disease names for legend
top_diseases_capitalized <- capitalize_first(top_diseases)

# 7-color palette for disease categories + Other
disease_palette <- c("#A01813", "#E3A772", "#2E5943", "#4F97CF", "#D65A1F", "#C9A020", "#359e80")
disease_colors <- c(
    setNames(disease_palette[seq_along(top_diseases)], top_diseases),
    Other = "#d9d9d9"
)

# Create labels for legend (capitalized)
disease_labels <- c(
    setNames(top_diseases_capitalized, top_diseases),
    Other = "Other"
)

# Assign disease group (top 7 or Other)
df_combined <- df_combined %>%
    mutate(
        diseaseGroup = if_else(diseaseNames %in% top_diseases, diseaseNames, "Other"),
        diseaseGroup = factor(diseaseGroup, levels = c(top_diseases, "Other"))
    )

# Create faceted scatter plot
p_c <- ggplot(df_combined, aes(
    x = estimatedBeta, y = neg_log10_p,
    colour = diseaseGroup
)) +
    geom_vline(xintercept = 0, colour = "#D65A1F", linetype = "dashed") +
    geom_point(size = 1, alpha = 0.85) +
    facet_grid(variant ~ ., scales = "free_y", axes = "all_x", axis.labels = "margins") +
    scale_colour_manual(
        values = disease_colors,
        breaks = c(top_diseases, "Other"),
        labels = disease_labels,
        name = NULL
    ) +
    scale_y_log10() +
    labs(
        tag = "c",
        x = expression("Estimated " * beta),
        y = "-log10(p-value)"
    ) +
    base_theme +
    all_text_8_theme() +
    theme(plot.margin = margin(0, 8, 0, 0, unit = "pt")) +
    theme(
        legend.position = c(0.60, 0.25),
        legend.justification = c(0, 0.5),
        legend.background = element_rect(fill = NA, colour = NA),
        legend.box.background = element_rect(fill = NA, colour = NA),
        legend.key = element_rect(fill = NA, colour = NA),
        legend.key.size = unit(0.4, "cm"),
        legend.key.width = unit(0.4, "cm"),
        legend.text = element_text(size = legend_text_size, color = text_colour),
        legend.spacing.y = unit(0.01, "cm"),
        plot.tag = element_text(face = "bold", size = 7, colour = text_colour),
        plot.tag.position = "topleft",
        axis.title.x = element_text(margin = margin(t = 8)),
        strip.text.y = element_text(size = 7, color = text_colour, angle = 270),
        strip.placement = "outside",
        panel.spacing.y = unit(0, "cm"),
        panel.border = element_blank()
    )

# -----------------------------
# Combine and save
# -----------------------------
out_path <- file.path(figure_3_dir, "figure_3_final.png")
if (!nzchar(figure_3_dir)) out_path <- "figure_3_final.png"

if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Please install 'patchwork' to combine the plots into a grid.")
}
suppressPackageStartupMessages(library(patchwork))

# Grid design: A (top-left), B (bottom-left), C spans full right column
# Patchwork aligns panels in the same row, so B and C bottom share baseline
design <- "
AC
BC
"
combined <- p_b + p_a + p_c +
    plot_layout(design = design, widths = c(1, 4), heights = c(1, 1))

# Save at A4 portrait width
aspect_ratio <- 3 / 7
a4_portrait_width_in <- 8.27
ggsave(out_path, combined, width = a4_portrait_width_in, height = a4_portrait_width_in * aspect_ratio, units = "in", dpi = 300, bg = "white")
message("Saved: ", out_path)
