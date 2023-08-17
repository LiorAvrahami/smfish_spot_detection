# x,y,z
CROPPED_IMAGE_SIZE = (100, 100, 10)

NORMALIZATION_QUANTILES = (0.8, 0.95)
# the default relevant channels
DEFAULT_CHANNELS = (1, 2, 3)
# the default Z stack below which we throw away
DEFAULT_ZMIN = 3
# conversion between micro meters to pixels
CONVERSION_FACTOR_UM_TO_PIXELS = 0.11