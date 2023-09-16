# x,y,z
CROPPED_IMAGE_SIZE = (100, 100)

NORMALIZATION_QUANTILES = (0.8, 0.95)
# conversion between micro meters to pixels
CONVERSION_FACTOR_UM_TO_PIXELS = 0.11
# files that only make life harder
FILE_EXTENTIONS_TO_IGNORE = [".db"]
# the maximum distances over each axis a spot can be from an roi's center for that spot to be deemed "inside the roi" (x,y,z)
# for example if the MAX_DISTANCE_FROM_ROI_CENTER is (1,1,0.5) following displacement vectors do not correspond to points that are inside the roi:
# (2,0,0), (0.8,0.8,0), (0,0,0.6), (0.6,0.6,0.3)
MAX_DISTANCE_FROM_ROI_CENTER = (2.5, 2.5, 1.2)
