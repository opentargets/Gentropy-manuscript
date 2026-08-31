suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

# Run from the repository root: tools/run_r.sh chapters/04-figures-main/figure_1/Figure_1_d_pychart.R
data_dir <- "data/intermediate_files_refactor"
fig1_dir <- "chapters/04-figures-main/figure_1"

# Slice counts come from chapters/02-analysis-main/01_panoramic.ipynb rather than being hardcoded
# here. Five slices: EUR and mixed are the reclassified ancestry groups, AFR and EAS/CSA are the
# non-EUR studies whose predominant ancestry is afr and eas, and Other is what remains
# (predominantly amr, plus Finnish). The published donut had four slices and buried all 11,725
# pan-ancestry studies inside Other.
donut_csv <- file.path(data_dir, "fig1d_ancestry_donut.csv")
stopifnot(file.exists(donut_csv))
counts <- suppressMessages(readr::read_csv(donut_csv, show_col_types = FALSE))

slice_levels <- c("EUR", "AFR", "EAS/CSA", "mixed", "Other")
stopifnot(setequal(counts$ancestry, slice_levels))

# Data frame
df <- counts %>%
  mutate(ancestry = factor(.data$ancestry, levels = slice_levels)) %>%
  arrange(.data$ancestry) %>%
  mutate(
    total = sum(.data$value),
    fraction = .data$value / .data$total,
    percent = .data$fraction * 100,
    label = paste0(.data$ancestry, "\n", round(.data$percent), "%"),
    y_mid = cumsum(.data$fraction) - .data$fraction / 2
  )

# Color map
# `mixed` gets the mid-blue used for the non-EUR layer of panel c rather than another step on the
# green ramp: it is not a single ancestry, and it must not read as one.
fill_colors <- c(
  "EUR" = "#2E5943",
  "AFR" = "#528B78",
  "EAS/CSA" = "#9EBAA8",
  "mixed" = "#8ABADE",
  "Other" = "lightgrey"
)

text_colors <- c(
  "EUR" = "#ffffff",
  "AFR" = "#ffffff",
  "EAS/CSA" = "#ffffff",
  "mixed" = "#434343",
  "Other" = "#434343"
)

# Donut chart
p <- ggplot(df, aes(x = 2, y = fraction, fill = ancestry)) +
  geom_col(width = 1, color = "white", linewidth = 2) +
  coord_polar(theta = "y", start = (pi / 2) - (90 * pi / 180)) +
  scale_fill_manual(values = fill_colors) +
  theme_void(base_family = "Helvetica") +
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "#ffffff00", color = NA)
  ) +
  xlim(c(0.5, 2.5)) +
  geom_text(
    aes(x = 2, label = label, color = ancestry),
    position = position_stack(vjust = 0.5),
    fontface = "bold",
    size = 3
  ) +
  scale_color_manual(values = text_colors, guide = "none")


# Standalone output. Figure_1_combined.R truncates this script at the first ggsave() line and keeps
# only the ggplot object `p`, so nothing below runs in the combined pipeline.
ggsave(filename = file.path(fig1_dir, "ancestry_donut.png"), plot = p,
       width = 4.5, height = 4.5, dpi = 300, bg = "#ffffff00")

# Print to viewer if interactive
# if (interactive()) print(p)
