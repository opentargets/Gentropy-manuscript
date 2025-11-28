suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# Input counts (match notebook values)
n_eur <- 65222L
n_afr <- 7421L
n_eas <- 11785L
n_other <- 16098L

# Data frame
df <- tibble::tibble(
  ancestry = factor(c("EUR", "AFR", "EAS/CSA", "Other"),
                    levels = c("EUR", "AFR", "EAS/CSA", "Other")),
  value = c(n_eur, n_afr, n_eas, n_other)
) %>%
  mutate(
    total = sum(value),
    fraction = value / total,
    percent = fraction * 100,
    label = paste0(ancestry, "\n", round(percent), "%"),
    label_color = ifelse(ancestry == "Other", "black", "white"),
    y_mid = cumsum(fraction) - fraction / 2
  )

# Color map
fill_colors <- c(
  "EUR" = "#2E5943",
  "AFR" = "#528B78",
  "EAS/CSA" = "#9EBAA8",
  "Other" = "lightgrey"
)

text_colors <- c(
  "EUR" = "#ffffff",
  "AFR" = "#ffffff",
  "EAS/CSA" = "#ffffff",
  "Other" = "#434343"
)

# Donut chart
p <- ggplot(df, aes(x = 2, y = fraction, fill = ancestry)) +
  geom_col(width = 1, color = "white", size = 2) +
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
    size = 4
  ) +
  scale_color_manual(values = text_colors, guide = "none")

# No center text

# Output directory (kept tidy within this experiment folder)
out_dir <- file.path("scr", "paper_figs", "pychart_R", "out")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Save outputs (4.5 x 4.5 inches, white background)
ggsave(filename = file.path(out_dir, "ancestry_donut.png"), plot = p,
       width = 4.5, height = 4.5, dpi = 300, bg = "#ffffff00")
# ggsave(filename = file.path(out_dir, "ancestry_donut.pdf"), plot = p,
#        width = 4.5, height = 4.5, bg = "white", device = "pdf")

# Print to viewer if interactive
# if (interactive()) print(p)


