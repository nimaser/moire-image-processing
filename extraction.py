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

### APPROACH 1 - amplitude ###

med = spnd.median_filter(img, 3)

smoothed1 = spnd.gaussian_filter(med, 1.3)
smoothed2 = spnd.gaussian_filter(smoothed1, 1.3)

dilated = spnd.maximum_filter(smoothed2, 15)

maxes = np.where(smoothed2 == dilated, med, np.zeros(img.shape))

border = ut.get_border_mask(np.array(maxes.shape), 1)
noborder = np.where(border, maxes, np.zeros(img.shape))

max_indices = np.argwhere(noborder > 0)

# plot everything
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8), layout='constrained')
fig.suptitle("amplitude method")
mc = MultiCursor(None, axs.flatten(), True, True, True, color='b', lw=1)
axs[0, 0].imshow(img       , cmap="gray")
axs[0, 1].imshow(med       , cmap="gray")
axs[0, 2].imshow(smoothed1 , cmap="gray")
axs[1, 0].imshow(smoothed2 , cmap="gray")
axs[1, 1].imshow(dilated   , cmap="gray")
axs[1, 2].imshow(noborder  , cmap="gray")
axs[0, 0].set_title("original")
axs[0, 1].set_title("median filtered")
axs[0, 2].set_title("smoothed once")
axs[1, 0].set_title("smoothed twice")
axs[1, 1].set_title("maximum filter")
axs[1, 2].set_title("maxes (excluding any at the border)")

for row, col in max_indices:
    for ax in axs.flatten():
        # x is cols, y is rows
        ax.add_patch(Circle((col, row), 3, color='b', fill=False))

plt.show()

### APPROACH 2 - shape ###

# binarize image
_, binary = cv2.threshold(img, 0, np.max(img), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
binary[binary != 0] = 1

# apply a floodfill to the image to fill the outside of the shapes we want to fill in
# invert the filled image to get the negative of the original binary image (the holes)
holes = binary.copy()
cv2.floodFill(holes, None, (binary.shape[0], 0), 1)
holes = np.logical_not(holes)

filled = np.logical_or(binary, holes)

# smooth out the jagged edges
k = ut.get_circular_kernel(9)
circular = spnd.median_filter(filled, footprint=k)

# remove any blobs attached to the edges
final_blobs = ut.clean_edges_binary(circular)

# get blob centroids
centroids = ut.get_blob_centroids(final_blobs)

# set up plot
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8), layout='constrained')
fig.suptitle("shape method")
mc = MultiCursor(None, axs.flatten(), True, True, True, color='b', lw=1)
axs[0, 0].imshow(img         , cmap="gray")
axs[0, 1].imshow(binary      , cmap="gray")
axs[0, 2].imshow(holes       , cmap="gray")
axs[1, 0].imshow(filled      , cmap="gray")
axs[1, 1].imshow(circular    , cmap="gray")
axs[1, 2].imshow(final_blobs , cmap="gray")
axs[0, 0].set_title("original")
axs[0, 1].set_title("thresholded to binary")
axs[0, 2].set_title("holes inside closed shapes")
axs[1, 0].set_title("filled in closed shapes")
axs[1, 1].set_title("median filter")
axs[1, 2].set_title("removing blobs attached to edges")

for centroid_row, centroid_col in centroids:
    for ax in axs.flatten():
        # x is cols, y is rows
        ax.add_patch(Circle((centroid_col, centroid_row), 3, color='r', fill=False))
	
plt.show()

### APPROACH 3 - hough ###



### COMPARISON ###

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 8), layout='constrained')
fig.suptitle("amplitude vs shape methods")
mc = MultiCursor(None, axs.flatten(), True, True, True, color='b', lw=1)
axs[0].imshow(img         , cmap="gray")
axs[1].imshow(noborder    , cmap="gray")
axs[2].imshow(final_blobs , cmap="gray")
axs[0].set_title("original")
axs[1].set_title("amplitude method")
axs[2].set_title("shape method")

for centroid_row, centroid_col in centroids:
    for ax in axs.flatten():
        # x is cols, y is rows
        ax.add_patch(Circle((centroid_col, centroid_row), 3, color='b', fill=False))
        
for row, col in max_indices:
    for ax in axs.flatten():
        # x is cols, y is rows
        ax.add_patch(Circle((col, row), 3, color='r', fill=False))
	
plt.show()