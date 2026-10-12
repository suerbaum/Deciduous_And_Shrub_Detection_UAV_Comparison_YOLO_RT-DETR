library(lidR)

las9_1 <- readLAS(
  "/data/projects/Students/Master/Annika/thesis/drone_data/summer/processed/odm_georeferencing_9_part1/odm_georeferencing/odm_georeferenced_model.laz",
  select = "xyzc"
)
las9_2 <- readLAS(
  "/data/projects/Students/Master/Annika/thesis/drone_data/summer/processed/odm_georeferencing_9_part2/odm_georeferencing/odm_georeferenced_model.laz",
  select = "xyzc"
)
las9 <- rbind(las9_1, las9_2)

# try CSF with parameters tuned for open rangeland vegetation
las9 <- classify_ground(las9, algorithm = csf(
  sloop_smooth    = TRUE,   # smooth cloth on slopes — important for Ranch 9
  class_threshold = 0.5,
  cloth_resolution = 1.0,   # 1m cloth resolution
  rigidness       = 2,      # stiffer cloth — better for slopes
  iterations      = 500,
  time_step       = 0.65
))

# check what percentage was classified as ground
tab <- table(las9$Classification)
print(tab)
cat(paste("Ground percentage:", round(100*tab["2"]/sum(tab), 1), "%\n"))