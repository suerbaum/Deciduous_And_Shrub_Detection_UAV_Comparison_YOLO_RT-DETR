# ============================================================
# Annika - Master's Thesis
# CHM Detrending + Variogram Analysis
# Ranch 12
# ============================================================
# WHY DETRENDING INSTEAD OF DTM:
# CSF classified 78% of points as ground — far too high — because Ranch 12
# is almost entirely covered by continuous 30cm shrubs. The cloth simulation
# had no way to distinguish true ground from the shrub surface sitting just
# 30cm above it. The focal detrending approach sidesteps ground classification
# entirely: subtract a heavily smoothed DSM to approximate terrain trend.
# Residuals capture vegetation-scale height variation only.
#
# WHY DETRENDED DSM FOR VARIOGRAM (not raw CHM):
# The raw CHM variogram shows a range of ~300m, reflecting broad terrain
# slope across the ranch — not vegetation structure. The detrended DSM
# isolates vegetation-height variation. The variogram on the detrended DSM
# captures autocorrelation at vegetation scale (individual plant crowns),
# which is the relevant scale for CV design.
# ============================================================

library(terra)
library(gstat)
library(sf)
library(ggplot2)

# ============================================================
# OPTION A: LOAD ALREADY-PROCESSED RESULTS
# Run this block if restarting session — skips focal operations
# ============================================================
chm_01m_12          <- rast("C:/Master Physische Geographie/Canada/Masterarbeit/R/chm_ranch12_01m.tif")
dsm_detrended_12    <- rast("C:/Master Physische Geographie/Canada/Masterarbeit/R/dsm_detrended_10m_ranch12.tif")
dsm_detrended_20m_12 <- rast("C:/Master Physische Geographie/Canada/Masterarbeit/R/dsm_detrended_20m_ranch12.tif")

# ============================================================
# OPTION B: RECOMPUTE FROM SCRATCH
# ============================================================

# Load original CHM (1cm resolution from ODM)
#dsm_raw <- rast("C:/Master Physische Geographie/Canada/Masterarbeit/R/chm_ranch12.tif")

# Compute aggregation factor dynamically
#input_res <- res(dsm_raw)[1]
#cat("Input resolution:", input_res, "m\n")

#fact_01m <- round(0.1 / input_res)
#cat("Aggregation factor for 0.1m:", fact_01m, "\n")

# Aggregate to 0.1m
# 0.1m retains fine-scale vegetation structure (individual shrub crowns ~20-50cm)
#chm_01m_12 <- aggregate(dsm_raw, fact = fact_01m, fun = "mean", na.rm = TRUE)
cat("Resolution after aggregation to 0.1m:", res(chm_01m_12), "\n")

writeRaster(
  chm_01m_12,
  "C:/Master Physische Geographie/Canada/Masterarbeit/R/chm_ranch12_01m_v2.tif",
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "BIGTIFF=YES")
)
cat("Aggregated CHM saved.\n")

# ============================================================
# DETRENDING — 2 WINDOW SIZES AT 0.1m RESOLUTION
# Subtracts a smoothed version of the DSM to remove broad terrain trend.
# Residuals capture vegetation-scale height variation only.
# 10m window = 101x101 pixels at 0.1m resolution
# 20m window = 201x201 pixels at 0.1m resolution
# 10m window confirmed best in previous analysis:
#   median = -0.060, mean = 0.004 — nearly centered at 0
# NOTE: focal operations are slow — do not interrupt
# ============================================================

cat("Running 10m focal window...\n")
terrain_approx_10m <- focal(
  chm_01m_12,
  w     = matrix(1, 101, 101),
  fun   = "mean",
  na.rm = TRUE
)
dsm_detrended_12 <- chm_01m_12 - terrain_approx_10m
writeRaster(
  dsm_detrended_12,
  "C:/Master Physische Geographie/Canada/Masterarbeit/R/dsm_detrended_10m_ranch12.tif",
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "BIGTIFF=YES")
)
cat("10m detrended raster saved.\n")

cat("Running 20m focal window...\n")
terrain_approx_20m <- focal(
  chm_01m_12,
  w     = matrix(1, 201, 201),
  fun   = "mean",
  na.rm = TRUE
)
dsm_detrended_20m_12 <- chm_01m_12 - terrain_approx_20m
writeRaster(
  dsm_detrended_20m_12,
  "C:/Master Physische Geographie/Canada/Masterarbeit/R/dsm_detrended_20m_ranch12.tif",
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "BIGTIFF=YES")
)
cat("20m detrended raster saved.\n")

# ============================================================
# INSPECT DETRENDING RESULTS
# Mean should be near 0 — confirms terrain trend was removed
# Positive values = vegetation above terrain trend
# Negative values = terrain depressions
# If mean is still ~1300+ the subtraction failed
# terra::global() used instead of summary(values()) —
# values() loads entire raster into RAM, unsafe for large files
# ============================================================

par(mfrow = c(1, 2))
plot(dsm_detrended_12,     main = "Detrended DSM — 10m window Ranch 12")
plot(dsm_detrended_20m_12, main = "Detrended DSM — 20m window Ranch 12")
par(mfrow = c(1, 1))

cat("=== 10m window ===\n")
print(terra::global(dsm_detrended_12,     fun = c("min", "max", "mean"), na.rm = TRUE))
cat("=== 20m window ===\n")
print(terra::global(dsm_detrended_20m_12, fun = c("min", "max", "mean"), na.rm = TRUE))

# NA check — NAs appear at raster edges due to focal window
total_pixels <- ncell(dsm_detrended_12)
na_pixels    <- sum(is.na(values(dsm_detrended_12)))
cat("NA percentage:", round(na_pixels / total_pixels * 100, 1), "%\n")

na_mask <- is.na(dsm_detrended_12)
plot(na_mask, main = "NA locations Ranch 12 (yellow = NA, purple = valid)")

# ============================================================
# VARIOGRAM ANALYSIS — THREE SCALES
#
# All variograms run on the 10m detrended DSM (dsm_detrended_12).
# The 20m detrended DSM is not re-analysed here — the 10m window
# was confirmed best and is the one used for CNN input and CV design.
#
# Coarse (500m, 25m bins):
#   Shows ranch-scale trend. Expected: no sill within ranch extent.
#   Confirms spatial block CV is infeasible regardless of block size —
#   blocks would need to exceed ranch width (~500m) to achieve independence.
#
# Fine (30m, 1m bins):
#   Reveals vegetation-scale decorrelation.
#   Result: sill at ~6m — consistent with individual shrub crown diameter.
#   This is the key result for random CV justification.
#
# Medium (100m, 1m bins):
#   Confirms fine-scale sill holds beyond 30m.
#   Ensures the 6m sill is real and not a cutoff artifact.
#   Result: confirmed flat from ~6m to 100m.
# ============================================================

set.seed(42)
pts_12    <- spatSample(dsm_detrended_12, size = 50000, method = "random",
                        as.points = TRUE, na.rm = TRUE)
pts_sf_12 <- st_as_sf(pts_12)

# ── CRITICAL: ensure column is named Z for variogram formula ─────────────────
# spatSample names the column after the raster layer name, not guaranteed to be Z.
# Rename the first (values) column to Z before running any variogram.
names(pts_sf_12)[1] <- "Z"
cat("Column names in pts_sf_12:", names(pts_sf_12), "\n")  # should show: Z geometry

# ── COARSE VARIOGRAM ──────────────────────────────────────────────────────────
v_12 <- variogram(Z ~ 1, data = pts_sf_12, cutoff = 500, width = 25)
print(v_12)

# ── FINE VARIOGRAM ────────────────────────────────────────────────────────────
v_12_fine <- variogram(Z ~ 1, data = pts_sf_12, cutoff = 30, width = 1)
print(v_12_fine)

# ── MEDIUM VARIOGRAM ──────────────────────────────────────────────────────────
v_12_medium <- variogram(Z ~ 1, data = pts_sf_12, cutoff = 100, width = 1)
print(v_12_medium)

# ============================================================
# PLOTS
# ============================================================

# Coarse
ggplot(v_12, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 12 — coarse scale (25m bins, 500m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch12_coarse.png", width = 8, height = 5, dpi = 300)

# Fine
ggplot(v_12_fine, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 12 — fine scale (1m bins, 30m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()
write.csv(v_12_fine,
          "C:/Master Physische Geographie/Canada/Masterarbeit/R/variogram_ranch12_data.csv",
          row.names = FALSE)
ggsave("variogram_ranch12_fine.png", width = 8, height = 5, dpi = 300)

# Medium
ggplot(v_12_medium, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 12 — medium scale (1m bins, 100m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch12_medium.png", width = 8, height = 5, dpi = 300)

# ============================================================
# CONCLUSIONS — Ranch 12
#
# Coarse variogram: no sill within 500m ranch extent.
#   → Spatial block CV impossible: blocks would need to exceed ranch width.
#   → Drop at ~337-500m is a finite-area artifact (few pairs at ranch edges),
#     not a real sill.
#
# Fine variogram: sill at ~5-6m.
#   → Vegetation-height autocorrelation decorrelates at individual plant
#     crown scale (~20-50cm shrubs have ~0.3-0.5m radius → gap between
#     plants ~1-5m → sill at ~5-6m is ecologically consistent).
#
# Medium variogram: flat from ~6m to 100m — sill confirmed.
#
# Conservative decorrelation distance: 6m (maximum across all ranches).
# This supports random CV between individually labeled plants:
# plants separated by >6m are spatially independent at vegetation scale.
#
# Cross-ranch summary:
#   Ranch 12: sill ~6m (confirmed)
#   Ranch 9:  sill ~5m (confirmed to 100m)
#   Ranch 5:  run ranch5_variogram_clean.R to confirm
# ============================================================
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
#dsm_detrended_20m_5 <- rast("D:/thesis/resolution1/orthomosaics/dsm_detrended_20m_ranch5.tif")

# ============================================================
# OPTION B: PROCESS FROM SCRATCH
# ============================================================

chm_01m_5 <- rast("D:/thesis/resolution1/orthomosaics/chm_ranch5_10cm.tif")

cat("Resolution:", res(chm_01m_5), "\n")         # should be 0.1 0.1
cat("Extent:", as.vector(ext(chm_01m_5)), "\n")
cat("CRS:", crs(chm_01m_5, describe = TRUE)$code, "\n")  # should be 32611

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
# ============================================================
# Annika - Master's Thesis
# CHM Detrending + Variogram Analysis
# Ranch 9
# ============================================================
# CHM was derived from two ODM .laz point cloud files using LAScatalog
# (rbind failed due to RAM — 180M points across two ~250MB files).
# Output: chm_ranch9_10cm.tif at 0.1m resolution, EPSG:32611.
# Z values are absolute elevation (1325–1378m ASL) — not yet normalized.
# Detrending below removes terrain trend to isolate vegetation heights.
# ============================================================

library(terra)
library(blockCV)
library(gstat)
library(sf)
library(ggplot2)

# ============================================================
# OPTION A: LOAD ALREADY-PROCESSED RESULTS (run this if restarting session)
# ============================================================
chm_01m_9           <- rast("D:/thesis/resolution1/orthomosaics/chm_ranch9_10cm.tif")
dsm_detrended_9     <- rast("D:/thesis/resolution1/orthomosaics/dsm_detrended_10m_ranch9.tif")
dsm_detrended_20m_9 <- rast("D:/thesis/resolution1/orthomosaics/dsm_detrended_20m_ranch9.tif")

# ============================================================
# OPTION B: PROCESS FROM SCRATCH — start here
# ============================================================

chm_01m_9 <- rast("D:/thesis/resolution1/orthomosaics/chm_ranch9_10cm.tif")

cat("Resolution:", res(chm_01m_9), "\n")
cat("Extent:", as.vector(ext(chm_01m_9)), "\n")
cat("CRS:", crs(chm_01m_9, describe = TRUE)$code, "\n")

# ============================================================
# DETRENDING — 2 WINDOW SIZES AT 0.1m RESOLUTION
# 10m window = 101x101 pixels at 0.1m resolution
# 20m window = 201x201 pixels at 0.1m resolution
# NOTE: focal operations are slow — do not interrupt
# ============================================================

cat("Running 10m focal window...\n")
terrain_approx_10m <- focal(
  chm_01m_9,
  w     = matrix(1, 101, 101),
  fun   = "mean",
  na.rm = TRUE
)
dsm_detrended_9 <- chm_01m_9 - terrain_approx_10m
writeRaster(
  dsm_detrended_9,
  "D:/thesis/resolution1/orthomosaics/dsm_detrended_10m_ranch9.tif",
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "BIGTIFF=YES")
)
cat("10m detrended raster saved.\n")

cat("Running 20m focal window...\n")
terrain_approx_20m <- focal(
  chm_01m_9,
  w     = matrix(1, 201, 201),
  fun   = "mean",
  na.rm = TRUE
)
dsm_detrended_20m_9 <- chm_01m_9 - terrain_approx_20m
writeRaster(
  dsm_detrended_20m_9,
  "D:/thesis/resolution1/orthomosaics/dsm_detrended_20m_ranch9.tif",
  overwrite = TRUE,
  gdal = c("COMPRESS=LZW", "BIGTIFF=YES")
)
cat("20m detrended raster saved.\n")

# ============================================================
# INSPECT DETRENDING RESULTS
# Mean should be near 0 — confirms terrain trend was removed
# Max should be a few meters — realistic shrub/tree heights
# If mean is still ~1349 the subtraction failed
# ============================================================

par(mfrow = c(1, 2))
plot(dsm_detrended_9,     main = "Detrended DSM — 10m window Ranch 9")
plot(dsm_detrended_20m_9, main = "Detrended DSM — 20m window Ranch 9")
par(mfrow = c(1, 1))

cat("=== 10m window ===\n")
print(terra::global(dsm_detrended_9,     fun = c("min", "max", "mean"), na.rm = TRUE))
cat("=== 20m window ===\n")
print(terra::global(dsm_detrended_20m_9, fun = c("min", "max", "mean"), na.rm = TRUE))

total_pixels <- ncell(dsm_detrended_9)
na_pixels    <- sum(is.na(values(dsm_detrended_9)))
cat("NA percentage:", round(na_pixels / total_pixels * 100, 1), "%\n")

na_mask <- is.na(dsm_detrended_9)
plot(na_mask, main = "NA locations (yellow = NA, purple = valid)")

# ============================================================
# VARIOGRAM ANALYSIS
# Three scales:
#   coarse (500m, 25m bins) — shows ranch-scale trend, confirms no sill
#   fine   (30m,   1m bins) — reveals vegetation-scale decorrelation
#   medium (100m,  1m bins) — confirms fine-scale sill holds beyond 30m
# ============================================================

set.seed(42)
pts_9    <- spatSample(dsm_detrended_9, size = 50000, method = "random",
                       as.points = TRUE, na.rm = TRUE)
pts_sf_9 <- st_as_sf(pts_9)

# Coarse
v_9 <- variogram(Z ~ 1, data = pts_sf_9, cutoff = 500, width = 25)
print(v_9)

# Fine
v_9_fine <- variogram(Z ~ 1, data = pts_sf_9, cutoff = 30, width = 1)
print(v_9_fine)

# Medium
v_9_medium <- variogram(Z ~ 1, data = pts_sf_9, cutoff = 100, width = 1)
print(v_9_medium)

# ============================================================
# PLOTS
# ============================================================

# Coarse variogram
ggplot(v_9, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 9 — coarse scale (25m bins)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch9_coarse.png", width = 8, height = 5, dpi = 300)

# Fine variogram
ggplot(v_9_fine, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 9 — fine scale (1m bins, 30m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()
write.csv(v_9_fine,
          "C:/Master Physische Geographie/Canada/Masterarbeit/R/variogram_ranch9_data.csv",
          row.names = FALSE)
ggsave("variogram_ranch9_fine.png", width = 8, height = 5, dpi = 300)

# Medium variogram
ggplot(v_9_medium, aes(x = dist, y = gamma)) +
  geom_point(aes(size = np / 1e6)) +
  geom_line() +
  labs(
    title = "Variogram detrended DSM Ranch 9 — medium scale (1m bins, 100m cutoff)",
    x     = "Distance (m)",
    y     = "Semivariance",
    size  = "Pairs (millions)"
  ) +
  theme_minimal()

ggsave("variogram_ranch9_medium.png", width = 8, height = 5, dpi = 300)

# ============================================================
# CONCLUSION (update after inspecting plots):
# Fine-scale variogram expected to show sill at ~4-5m
# consistent with Ranch 12 (~6m) and Ranch 5 (~5m).
# Conservative decorrelation distance across all ranches: 6m.
# Coarse variogram expected to show no sill within ranch extent,
# confirming spatial block CV is infeasible regardless of block size.
# ============================================================