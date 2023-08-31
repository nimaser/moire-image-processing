# @author Nikhil Maserang
# @date 2023/08/14

import numpy as np
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import PIL.Image
import cv2
import utils as ut

# 002, 004, 005, 006, 008

### LOAD AND PLOT IMAGE DATA ###

fig, ax = plt.subplots(figsize=(12, 8))
mc = MultiCursor(None, [ax], horizOn=True, color='b', lw=1)

# get image data
fname = "images/m008.sxm"
if fname.endswith(".sxm"): img = ut.get_sxm_data(fname, False)
if fname.endswith(".jpg"): img = PIL.Image.open(fname)

# proportionally scale up data
scaledimg = ut.proportional_scale(img)

# shift data to all be > 0 if negative values are present
if np.min(scaledimg) < 0:
    shiftedimg = scaledimg - np.min(scaledimg)
else:
    shiftedimg = scaledimg
    
# convert to uints
data = shiftedimg.astype(np.uint16)

ut.add_processing_sequence(fig, ax, True, img, scaledimg, shiftedimg, data)
plt.show()

### APPROACH 1 - shape ###

# binarize image
_, binary = cv2.threshold(img, 0, np.max(img), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
binary[binary != 0] = 1

# fill in the holes
print(binary.shape)
filled = ut.fill_holes(binary, (0, 0))

# smooth out the jagged edges with a median filter to get blobs
k = ut.get_circular_kernel(9)
blobs = spnd.median_filter(filled, footprint=k)

# remove any blobs attached to the edges
final_blobs = ut.clean_edges_binary(blobs)

# get blob centroids
centroids = ut.get_blob_centroids(final_blobs)

# set up plot
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8), layout='constrained')
fig.suptitle("Approach 1: blob method\nPress 1 to toggle circles")
mc = MultiCursor(None, axs.flatten(), horizOn=True, color='b', lw=1)
axs[0, 0].imshow(img         , cmap="gray")
axs[0, 1].imshow(binary      , cmap="gray")
axs[0, 2].imshow(filled      , cmap="gray")
axs[1, 0].imshow(blobs       , cmap="gray")
axs[1, 1].imshow(final_blobs , cmap="gray")
ut.add_processing_sequence(fig, axs[1, 2], False, img, binary, filled, blobs, final_blobs)
axs[0, 0].set_title("original img")
axs[0, 1].set_title("binarize via thresholding")
axs[0, 2].set_title("fill in closed shapes")
axs[1, 0].set_title("smooth edges and make blobs via median filter")
axs[1, 1].set_title("remove blobs attached to edges")
axs[1, 2].set_title("press < , . > to scroll through")

if len(centroids.shape) > 1:
    ut.add_toggleable_circles(fig, axs, np.roll(centroids, 1, axis=1), '1')

plt.show()

### APPROACH 2 - amplitude ###

med = spnd.median_filter(img, 3)

smoothed1 = spnd.gaussian_filter(med, 1.3)
smoothed2 = spnd.gaussian_filter(smoothed1, 1.3)

dilated = spnd.maximum_filter(smoothed2, 15)

maxes = np.where(smoothed2 == dilated, med, np.zeros(img.shape))

# plot everything
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8), layout='constrained')
fig.suptitle("amplitude method")
mc = MultiCursor(None, axs.flatten(), True, True, True, color='b', lw=1)
axs[0, 0].imshow(img       , cmap="gray")
axs[0, 1].imshow(med       , cmap="gray")
axs[0, 2].imshow(smoothed1 , cmap="gray")
axs[1, 0].imshow(smoothed2 , cmap="gray")
axs[1, 1].imshow(dilated   , cmap="gray")
axs[1, 2].imshow(maxes  , cmap="gray")
axs[0, 0].set_title("original")
axs[0, 1].set_title("median filtered")
axs[0, 2].set_title("smoothed once")
axs[1, 0].set_title("smoothed twice")
axs[1, 1].set_title("maximum filter")
axs[1, 2].set_title("maxes")

ut.add_toggleable_circles(fig, axs, np.roll(centroids, 1, axis=1), '1')

plt.show()

### APPROACH 3 - combined shape + amplitude ###

# get mask for areas around the centers computed via the shape method
confirmation_mask = np.zeros(maxes.shape)
for center in centroids:
    ut.set_overlay_value(confirmation_mask, center, 7, 1)

confirmed = maxes & confirmation_mask

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8), layout='constrained')
fig.suptitle("combined method")
mc = MultiCursor(None, axs.flatten(), True, True, True, color='b', lw=1)
axs[0, 0].imshow(img, cmap="gray")
axs[0, 1].imshow(maxes, cmap="gray")
axs[1, 0].imshow(confirmed, cmap="gray")
axs[1, 1].imshow(confirmed, cmap="gray")



### APPROACH 4 - hough


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