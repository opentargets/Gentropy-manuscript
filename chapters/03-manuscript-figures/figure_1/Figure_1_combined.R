# Figure_1_combined.R  — fully vector PDF
#
# Combines the time-series facet panels (left) with the circular Manhattan
# plot (right) into a single vector PDF without rasterising either panel.
#
# Strategy:
#   Left  – source Figure_1_facet.R (skip ggsave) → combined_grob drawn via grid
#   Right – source manh_plot.R (skip main()) → call create_circular_manhattan()
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

fig1_dir          <- "/Users/polina/Gentropy-manuscript/chapters/03-manuscript-figures/figure_1"
facet_script_path <- file.path(fig1_dir, "Figure_1_facet.R")
manh_script_path  <- file.path(fig1_dir, "manh_plot.R")
output_path       <- file.path(fig1_dir, "Figure_1_combined.pdf")
top_panel_pdf     <- file.path(fig1_dir, "Fig1 a (cropped).pdf")

# ── 1. Build facet grob from Figure_1_facet.R ────────────────────────────────
# Drop the final ggsave() so combined_grob stays in memory
facet_lines <- readLines(facet_script_path)
save_idx    <- tail(grep("^ggsave\\(", facet_lines), 1)
if (length(save_idx)) facet_lines <- facet_lines[-save_idx]
eval(parse(text = paste(facet_lines, collapse = "\n")), envir = environment())
# combined_grob is now in scope

# ── 2. Load circular Manhattan functions from manh_plot.R ────────────────────
# Drop the final main() call so only functions are defined, not executed
manh_lines    <- readLines(manh_script_path)
main_call_idx <- tail(grep("^main\\(\\)$", manh_lines), 1)
if (length(main_call_idx)) manh_lines <- manh_lines[-main_call_idx]
eval(parse(text = paste(manh_lines, collapse = "\n")), envir = environment())
# create_circular_manhattan(), read_parquet_data(), etc. are now available

# Load the data (path from manh_plot.R's main())
parquet_file <- paste0(
  "/Users/polina/genetics_gsea/data/disease_ta_measur_index/",
  "part-00000-6aad212f-e927-4ad8-8e92-57687d88f801-c000.snappy.parquet"
)
cat("Loading parquet data...\n")
circo_data <- read_parquet_data(parquet_file)

# ── 3. Compose the combined PDF ───────────────────────────────────────────────
# Independent size controls:
#   left_w   – width  of Plot A (inches)
#   left_h   – height of Plot A (inches)  ← change this freely
#   total_w  – full device width; right column = total_w - left_w (B's square side)
#   left_w must equal 0.3 × total_w to keep right_x1_ndc at exactly 0.3
#
# Device height = tallest of A or B+pad; the shorter panel is centred vertically.
total_w      <- 16.5
left_w       <- 4.8          # = 0.3 × total_w  →  right_x1_ndc exactly 0.3
left_h       <- 11           # ← Plot A height (change independently)
a_top_margin <- 0.3          # ← white space above Plot A (inches)
right_sq     <- total_w - left_w       # B's square side
circ_pad     <- 0.5                    # inches below circle for gene labels
panels_h     <- max(left_h, right_sq + circ_pad)  # height of the A+B area

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

quartz(type = "pdf", file = output_path, width = total_w, height = total_h, bg = "white")

# — Right panel: circlize ————————————————————————————————————————————————
par(
  fig = c(right_x1_ndc, 1.0, bottom_pad_ndc, top_ndc),
  mar = c(0, 0, 0, 0),
  oma = c(0, 0, 0, 0)
)
create_circular_manhattan(circo_data, output_file = NULL)

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

# — Debug borders ————————————————————————————————————————————————————————
grid.rect(x=0, y=unit(a_y_off,"inches"),
  width=unit(right_x1_ndc,"npc"), height=unit(left_h,"inches"),
  just=c("left","bottom"), gp=gpar(col="#E74C3C",fill=NA,lwd=1.2,lty="dashed"))
grid.rect(x=unit(right_x1_ndc,"npc"), y=unit(bottom_pad_ndc,"npc"),
  width=unit(1-right_x1_ndc,"npc"), height=unit(top_ndc-bottom_pad_ndc,"npc"),
  just=c("left","bottom"), gp=gpar(col="#3498DB",fill=NA,lwd=1.2,lty="dashed"))
grid.rect(x=0, y=unit(panels_top_ndc,"npc"),
  width=unit(1,"npc"), height=unit(1-panels_top_ndc,"npc"),
  just=c("left","bottom"), gp=gpar(col="#27AE60",fill=NA,lwd=1.2,lty="dashed"))

# — Panel labels ——————————————————————————————————————————————————————————
lbl_gp <- gpar(fontsize = 12, fontface = "bold", col = "#434343")
grid.text("b", x=unit(4,"pt"),
  y=unit(a_y_off+left_h,"inches")-unit(4,"pt"), just=c("left","top"), gp=lbl_gp)
grid.text("d", x=unit(right_x1_ndc+0.005,"npc"),
  y=unit(top_ndc,"npc")-unit(4,"pt"), just=c("left","top"), gp=lbl_gp)

dev.off()
cat("PDF saved to:", output_path, "\n")
