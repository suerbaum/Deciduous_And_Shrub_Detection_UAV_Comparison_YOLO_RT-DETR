# ============================================================
# Annika - Master's Thesis
# CHM Detrending + Variogram Analysis
# Ranch 5
# ============================================================
# WHY DETRENDED DSM FOR VARIOGRAM:
# The raw CHM contains both terrain trend and vegetation signal.
# Running the variogram on the raw CHM produces a range of ~300m,
# reflecting broad terrain slope — not vegetation structure.
# The detrended DSM isolates vegetation-height variation by removing
# the terrain trend via focal smoothing. The variogram on the detrended
# DSM therefore captures autocorrelation at vegetation scale (individual
# plant crowns), which is the relevant scale for CV design.
# ============================================================

library(terra)
library(gstat)
library(sf)
library(ggplot2)

# ============================================================
# OPTION A: LOAD ALREADY-PROCESSED RESULTS
# Run this block if restarting session — skips focal operations
# ============================================================
chm_01m_5           <- rast("D:/thesis/resolution1/orthomosaics/chm_ranch5_10cm.tif")
dsm_detrended_5     <- rast("D:/thesis/resolution1/orthomosaics/dsm_detrended_10m_ranch5.tif")
dsm_detrended_20m_5 <- rast("D:/thesis/resolution1/orthomosaics/dsm_detrended_20m_ranch5.tif")

# ============================================================
# OPTION B: PROCESS FROM SCRATCH
# ============================================================

#chm_01m_5 <- rast("D:/thesis/resolution1/orthomosaics/chm_ranch5_10cm.tif")

#cat("Resolution:", res(chm_01m_5), "\n")         # should be 0.1 0.1
#cat("Extent:", as.vector(ext(chm_01m_5)), "\n")
#cat("CRS:", crs(chm_01m_5, describe = TRUE)$code, "\n")  # should be 32611

# ============================================================
# DETRENDING — 2 WINDOW SIZES AT 0.1m RESOLUTION
# Subtracts a smoothed version of the DSM to remove broad terrain trend.
# Residuals capture vegetation-scale height variation only.
# 10m window = 101x101 pixels at 0.1m resolution
# 20m window = 201x201 pixels at 0.1m resolution
# NOTE: focal operations are slow — do not interrupt
# ============================================================

cat("Running 10m focal window...\n")
terrain_approx_10m <- focal(
  chm_01m_5,
  w     = matrix(1, 101, 101),
  fun   = "mean",
  na.rm = TRUE
)
dsm_detrended_5 <- chm_01m_5 - terrain_approx_10m
writeRaster(
  dsm_detrended_5,
  "D:/thesis/resolution1/orthomosaics/dsm_detrended_10m_ranch5.tif",
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "BIGTIFF=YES")
)
cat("10m detrended raster saved.\n")
### # 20 meter in python
dsm_detrended_20m_5 <- rast(
  "D:/thesis/resolution1/orthomosaics/dsm_detrended_20m_ranch5.tif"
)

set.seed(42)
pts_5_20m    <- spatSample(dsm_detrended_20m_5, size = 50000, 
                           method = "random", as.points = TRUE, na.rm = TRUE)
pts_sf_5_20m <- st_as_sf(pts_5_20m)
names(pts_sf_5_20m)[1] <- "Z"

v_5_20m_fine <- variogram(Z ~ 1, data = pts_sf_5_20m, cutoff = 30, width = 1)
print(v_5_20m_fine)
# ============================================================
# INSPECT DETRENDING RESULTS
# Mean should be near 0 — confirms terrain trend was removed
# Positive values = vegetation above terrain trend
# Negative values = terrain depressions
# If mean is still ~1300+ the subtraction failed
# ============================================================

par(mfrow = c(1, 2))
plot(dsm_detrended_5,     main = "Detrended DSM — 10m window Ranch 5")
plot(dsm_detrended_20m_5, main = "Detrended DSM — 20m window Ranch 5")
par(mfrow = c(1, 1))

cat("=== 10m window ===\n")
print(terra::global(dsm_detrended_5,     fun = c("min", "max", "mean"), na.rm = TRUE))
cat("=== 20m window ===\n")
print(terra::global(dsm_detrended_20m_5, fun = c("min", "max", "mean"), na.rm = TRUE))

total_pixels <- ncell(dsm_detrended_5)
na_pixels    <- sum(is.na(values(dsm_detrended_5)))
cat("NA percentage:", round(na_pixels / total_pixels * 100, 1), "%\n")

na_mask <- is.na(dsm_detrended_5)
plot(na_mask, main = "NA locations Ranch 5 (yellow = NA, purple = valid)")

# ============================================================
# VARIOGRAM ANALYSIS — THREE SCALES
#
# Purpose: justify random CV by showing vegetation-height
# autocorrelation decorrelates at plant-crown scale (~5-6m).
#
# Coarse (500m, 25m bins):
#   Shows ranch-scale trend. Expected: no sill within ranch extent.
#   Confirms spatial block CV is infeasible — blocks would need to
#   exceed ranch width to achieve spatial independence.
#
# Fine (30m, 1m bins):
#   Reveals vegetation-scale decorrelation.
#   Expected: sill at ~4-6m, consistent with Ranches 9 and 12.
#   This is the key result for CV justification.
#
# Medium (100m, 1m bins):
#   Confirms the fine-scale sill holds beyond 30m.
#   Ensures sill is real and not a cutoff artifact.
# ============================================================

set.seed(42)
pts_5    <- spatSample(dsm_detrended_5, size = 50000, method = "random",
                       as.points = TRUE, na.rm = TRUE)
pts_sf_5 <- st_as_sf(pts_5)

# ── CRITICAL: ensure column is named Z for variogram formula ─────────────────
# spatSample names the column after the raster layer name, which may not be Z.
# Rename the first (values) column to Z regardless of what it is currently.
names(pts_sf_5)[1] <- "Z"
cat("Column names in pts_sf_5:", names(pts_sf_5), "\n")  # should show Z and geometry

# ── COARSE VARIOGRAM ──────────────────────────────────────────────────────────
v_5 <- variogram(Z ~ 1, data = pts_sf_5, cutoff = 500, width = 25)
print(v_5)

# ── FINE VARIOGRAM ────────────────────────────────────────────────────────────
v_5_fine <- variogram(Z ~ 1, data = pts_sf_5, cutoff = 30, width = 1)
print(v_5_fine)

# ── MEDIUM VARIOGRAM ──────────────────────────────────────────────────────────
v_5_medium <- variogram(Z ~ 1, data = pts_sf_5, cutoff = 100, width = 1)
print(v_5_medium)
####

# ============================================================
# PLOTS
# ============================================================

# Coarse
ggplot(v_5, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 5 — coarse scale (25m bins, 500m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch5_coarse.png", width = 8, height = 5, dpi = 300)

# Fine
ggplot(v_5_fine, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 5 — fine scale (1m bins, 30m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch5_fine.png", width = 8, height = 5, dpi = 300)

# Medium
ggplot(v_5_medium, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 5 — medium scale (1m bins, 100m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch5_medium.png", width = 8, height = 5, dpi = 300)


###
# ── LOAD 20m DETRENDED RASTER ─────────────────────────────────────────────────
dsm_detrended_20m_5 <- rast("D:/thesis/resolution1/orthomosaics/dsm_detrended_20m_ranch5.tif")

# ── SAMPLE ────────────────────────────────────────────────────────────────────
set.seed(42)
pts_5_20m    <- spatSample(dsm_detrended_20m_5, size = 50000, method = "random",
                           as.points = TRUE, na.rm = TRUE)
pts_sf_5_20m <- st_as_sf(pts_5_20m)
names(pts_sf_5_20m)[1] <- "Z"

# ── VARIOGRAMS ────────────────────────────────────────────────────────────────
v_5_20m        <- variogram(Z ~ 1, data = pts_sf_5_20m, cutoff = 500, width = 25)
v_5_20m_fine   <- variogram(Z ~ 1, data = pts_sf_5_20m, cutoff = 30,  width = 1)
v_5_20m_medium <- variogram(Z ~ 1, data = pts_sf_5_20m, cutoff = 100, width = 1)

print(v_5_20m_fine)
print(v_5_20m_medium)

# ── COMPARISON PLOT — 10m vs 20m detrending, fine scale ──────────────────────
# Adds a window column to each result then combines for side-by-side comparison
v_5_fine$window    <- "10m detrending"
v_5_20m_fine$window <- "20m detrending"
v_compare <- rbind(v_5_fine, v_5_20m_fine)

ggplot(v_compare, aes(x = dist, y = gamma, color = window)) +
  geom_line() +
  geom_point(aes(size = np / 1e6)) +
  labs(
    title = "Ranch 5 — fine scale variogram: 10m vs 20m detrending",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)",
    color = "Detrending window"
  ) +
  theme_minimal()
# export variogram data for Python figure
write.csv(v_5_fine, 
          "C:/Master Physische Geographie/Canada/Masterarbeit/R/variogram_ranch5_data.csv", 
          row.names = FALSE)
ggsave("variogram_ranch5_10m_vs_20m_fine.png", width = 9, height = 5, dpi = 300)
# ============================================================
# CROSS-RANCH SUMMARY — run after all three ranches are complete
# ============================================================

# Collect sill distances from fine variograms across all ranches
# Update these values after inspecting each fine-scale plot:
#   Ranch 12: sill at ~6m  (confirmed)
#   Ranch 9:  sill at ~5m  (confirmed to 100m)
#   Ranch 5:  sill at ~?m  (update after running this script)

# Conservative decorrelation distance = maximum sill across all ranches
# Use this value to:
#   1. Justify random CV in thesis methods
#   2. Set minimum inter-tile buffer in stratified fold assignment

cat("\n=== CROSS-RANCH VARIOGRAM SUMMARY ===\n")
cat("Ranch 12 fine-scale sill: ~6m\n")
cat("Ranch 9  fine-scale sill: ~5m\n")
cat("Ranch 5  fine-scale sill: inspect v_5_fine output above\n")
cat("Conservative decorrelation distance: 6m (update if Ranch 5 exceeds this)\n")