# Figure_1_combined.R  — fully vector PDF
#
# Combines the time-series facet panels (left) with the circular Manhattan
# plot (right) into a single vector PDF without rasterising either panel.
#
# Strategy:
#   Left  – source Figure_1_b_c.R (skip ggsave) → combined_grob drawn via grid
#   Right – source Figure_1_d.R (skip main()) → call create_circular_manhattan()
#            with output_file = NULL so it draws to the active PDF device,
#            constrained to the right column by par(fig = ...)
#
# Output: Figure_1_combined.pdf  (16 × 11.7 inches)

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr)
  library(ggplot2); library(scales); library(stringr); library(rlang)
  library(grid); library(gtable)
  library(circlize); library(arrow); library(RColorBrewer); library(png)
})

# Detect this script's directory so all paths stay relative
.argv     <- commandArgs(trailingOnly = FALSE)
.file_arg <- .argv[startsWith(.argv, "--file=")]
fig1_dir  <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
}

facet_script_path   <- file.path(fig1_dir, "Figure_1_b_c.R")
manh_script_path    <- file.path(fig1_dir, "Figure_1_d.R")
pychart_script_path <- file.path(fig1_dir, "Figure_1_d_pychart.R")
output_path         <- file.path(fig1_dir, "Figure_1_combined.pdf")
top_panel_pdf       <- file.path(fig1_dir, "assets", "Fig1 a (cropped).pdf")

# ── 1. Build facet grob from Figure_1_b_c.R ─────────────────────────────────
# Drop the final ggsave() so combined_grob stays in memory
facet_lines <- readLines(facet_script_path)
save_idx    <- tail(grep("^ggsave\\(", facet_lines), 1)
if (length(save_idx)) facet_lines <- facet_lines[-save_idx]
eval(parse(text = paste(facet_lines, collapse = "\n")), envir = environment())
# combined_grob is now in scope

# ── 2. Load circular Manhattan functions from Figure_1_d.R ───────────────────
# Drop the final main() call so only functions are defined, not executed
manh_lines    <- readLines(manh_script_path)
main_call_idx <- tail(grep("^main\\(\\)$", manh_lines), 1)
if (length(main_call_idx)) manh_lines <- manh_lines[-main_call_idx]
eval(parse(text = paste(manh_lines, collapse = "\n")), envir = environment())
# create_circular_manhattan(), read_parquet_data(), etc. are now available

# Load the data (path from Figure_1_d.R's main())
parquet_file <- file.path(fig1_dir, "data", "disease_ta_measur_index.snappy.parquet")
cat("Loading parquet data...\n")
circo_data <- read_parquet_data(parquet_file)

# ── 2b. Build donut chart grob from Figure_1_d_pychart.R ─────────────────────
# Skip the ggsave() so `p` (the ggplot object) stays in memory
pychart_lines  <- readLines(pychart_script_path)
gsave_idx      <- grep("^ggsave\\(", pychart_lines)
if (length(gsave_idx)) pychart_lines <- head(pychart_lines, gsave_idx[1] - 1)
eval(parse(text = paste(pychart_lines, collapse = "\n")), envir = environment())
# `p` (the donut ggplot) is now in scope

# ── 3. Compose the combined PDF ───────────────────────────────────────────────
# Independent size controls:
#   left_w   – width  of Plot A (inches)
#   left_h   – height of Plot A (inches)  ← change this freely
#   total_w  – full device width; right column = total_w - left_w (B's square side)
#   left_w must equal 0.3 × total_w to keep right_x1_ndc at exactly 0.3
#
# Device height = tallest of A or B+pad; the shorter panel is centred vertically.
total_w      <- 15
left_w       <- 4.8          # = 0.3 × total_w  →  right_x1_ndc exactly 0.3
a_top_margin <- 0.3          # ← white space above Plot A (inches)
right_sq     <- total_w - left_w       # B's square side
circ_pad     <- 0.1                    # inches below circle for gene labels
panels_h     <- right_sq + circ_pad   # A+B area height (B determines it)
left_h       <- panels_h - 0.75           # Plot A matches exactly → no bottom gap

# Top panel: derive height from PDF aspect ratio so nothing distorts
if (!requireNamespace("pdftools", quietly = TRUE)) install.packages("pdftools", repos = "https://cloud.r-project.org")
pdf_sz      <- pdftools::pdf_pagesize(top_panel_pdf)[1, ]   # width/height in pts
top_panel_h <- total_w * (pdf_sz$height / pdf_sz$width)     # fills full width exactly

total_h      <- panels_h + top_panel_h

right_x1_ndc <- left_w / total_w     # 0.3 exactly

# A and B are top-aligned within the panels_h area (top panel sits above them)
a_y_off  <- panels_h - left_h         # A flush to top of panels area
b_y_off  <- 0                         # B at bottom of panels area

bottom_pad_ndc <- (b_y_off + circ_pad) / total_h
top_ndc        <- (b_y_off + circ_pad + right_sq) / total_h
panels_top_ndc <- panels_h / total_h

# — Inject "c" label into combined_grob (mirrors "b": 4 pt from panel top) ──
# p_pairs is the 3rd grob in the stack; its first row in combined_grob starts
# immediately after grobs_list[[1]] (p_samples) and [[2]] (p_beta).
.c_first_row <- nrow(grobs_list[[1]]) + nrow(grobs_list[[2]]) + 1
combined_grob <- gtable::gtable_add_grob(
  combined_grob,
  grobs = grid::textGrob("c",
    x    = grid::unit(4, "pt"),
    y    = grid::unit(1, "npc") + grid::unit(12, "pt"),
    just = c("left", "top"),
    gp   = grid::gpar(fontsize = 12, fontface = "bold", col = "#434343")
  ),
  t = .c_first_row, b = .c_first_row, l = 1, r = ncol(combined_grob),
  clip = "off", name = "c-label"
)

quartz(type = "pdf", file = output_path, width = total_w, height = total_h, bg = "white")

# — Right panel: circlize ————————————————————————————————————————————————
par(
  fig = c(right_x1_ndc, 1.0, bottom_pad_ndc, top_ndc),
  mar = c(0, 0, 0, 0),
  oma = c(0, 0, 0, 0)
)
create_circular_manhattan(circo_data, output_file = NULL, center_plot = p)

# — Left panel: ggplot2 grob via grid ————————————————————————————————————
pushViewport(viewport(
  x      = 0,
  y      = unit(a_y_off, "inches"),
  width  = right_x1_ndc,
  height = unit(left_h - a_top_margin, "inches"),
  just   = c("left", "bottom"),
  clip   = "on"
))
grid.draw(combined_grob)
popViewport()

# — Top panel: rasterise via sips (macOS built-in) ————————————————————————
top_tmp <- tempfile(fileext = ".png")
system2("sips", args = c("-s", "format", "png", "-Z", "3000",
                          shQuote(top_panel_pdf), "--out", shQuote(top_tmp)),
        stdout = FALSE, stderr = FALSE)
top_raw <- png::readPNG(top_tmp)
file.remove(top_tmp)
# Explicit RGB→hex conversion so as.raster() works correctly
h <- dim(top_raw)[1]; w <- dim(top_raw)[2]
nc <- dim(top_raw)[3]
if (nc >= 4) {
  top_cols <- rgb(top_raw[,,1], top_raw[,,2], top_raw[,,3], top_raw[,,4])
} else {
  top_cols <- rgb(top_raw[,,1], top_raw[,,2], top_raw[,,3])
}
top_raster <- as.raster(matrix(top_cols, nrow = h, ncol = w))

pushViewport(viewport(
  x      = 0,
  y      = unit(panels_top_ndc, "npc"),
  width  = 1,
  height = unit(1 - panels_top_ndc, "npc"),
  just   = c("left", "bottom")
))
grid.raster(top_raster, width = unit(1, "npc"), height = unit(1, "npc"))
grid.text("a", x = unit(4, "pt"), y = unit(1, "npc") - unit(4, "pt"),
  just = c("left", "top"), gp = gpar(fontsize = 12, fontface = "bold", col = "#434343"))
popViewport()

# — Panel labels ——————————————————————————————————————————————————————————
lbl_gp <- gpar(fontsize = 12, fontface = "bold", col = "#434343")
grid.text("b", x=unit(4,"pt"),
  y=unit(a_y_off+left_h,"inches")-unit(4,"pt"), just=c("left","top"), gp=lbl_gp)
grid.text("d", x=unit(right_x1_ndc+0.005,"npc"),
  y=unit(top_ndc,"npc")-unit(4,"pt"), just=c("left","top"), gp=lbl_gp)

dev.off()
cat("PDF saved to:", output_path, "\n")
