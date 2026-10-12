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