# @author Nikhil Maserang
# @date 2023/08/14

import numpy as np
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.patches import Circle
import PIL.Image
import cv2
import utils as ut

# get image data
img = np.array(PIL.Image.open("moire.jpg").convert('L'))