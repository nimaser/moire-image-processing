# @author Nikhil Maserang
# @date 2023/09/02

import numpy as np
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, TextBox

import PIL.Image
import utils as ut

# 002, 004, 005, 006, 008

# get image data
fname = "images/m004.sxm"
if fname.endswith(".jpg"):
    img = PIL.Image.open(fname)
if fname.endswith(".sxm"):
    img = ut.get_sxm_data(fname, False)
    img = ut.scale_to_uint8(img)
    
###########################################################

# set default extraction parameters
params = dict(
    flip = 0,
    mode = "auto",
    trsh = 116,
    blck = 11,
    thrc = 2,
    init = (0,0),
    smsz = 9
)

# initial data processing
*imgs, centers = ut.extraction(img, **params)
titles = [
    "original",
    "flipped",
    "binarized",
    "filled in",
    "blobs smoothed",
    "edges cleaned"
]

# command handling
cmdhandler = ut.CommandProcessor()
cmdhandler.add_cmd("-flip", 1, lambda x: params.update(dict(flip=int(x))))
cmdhandler.add_cmd("-mode", 1, lambda x: params.update(dict(mode=x)))
cmdhandler.add_cmd("-trsh", 1, lambda x: params.update(dict(trsh=int(x))))
cmdhandler.add_cmd("-blck", 1, lambda x: params.update(dict(blck=int(x))))
cmdhandler.add_cmd("-thrc", 1, lambda x: params.update(dict(thrc=int(x))))
cmdhandler.add_cmd("-smsz", 1, lambda x: params.update(dict(smsz=int(x))))

def parse_init_tuple(init):
    init = tuple(map(int, init.replace('(', '').replace(')', '').split(',')))
    params.update(dict(init=init))
cmdhandler.add_cmd("-init", 1, parse_init_tuple)

def open_overlay_window():
    fig, ax = plt.subplots(figsize=(8, 8))
    ut.add_processing_sequence(fig, ax, True, imgs, titles)
    if len(centers.shape) > 1:
        ut.add_toggleable_circles(fig, [ax], np.roll(centers, 1, axis=1), 'v')
    fig.show()
cmdhandler.add_cmd("-overlay", 0, open_overlay_window)

# create figure
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
fig.subplots_adjust(bottom=0.15)
mc = MultiCursor(None, axs.flatten(), horizOn=True, color='b', lw=1)

# axes titles
axs[0, 0].set_title(titles[0])
axs[0, 1].set_title(titles[1])
axs[0, 2].set_title(titles[2])
axs[1, 0].set_title(titles[3])
axs[1, 1].set_title(titles[4])
axs[1, 2].set_title(titles[5])

# plot images on axes
def update_axes():
    axs[0, 0].imshow(imgs[0], cmap="gray")
    axs[0, 1].imshow(imgs[1], cmap="gray")
    axs[0, 2].imshow(imgs[2], cmap="gray")
    axs[1, 0].imshow(imgs[3], cmap="gray")
    axs[1, 1].imshow(imgs[4], cmap="gray")
    axs[1, 2].imshow(imgs[5], cmap="gray")
update_axes()

# create command textbox
textboxbbox = plt.axes([0.2, 0.025, 0.6, 0.05])
textbox = TextBox(textboxbbox, '>>>', initial="")

def on_cmd_submit(text : str):
    cmdhandler.process_cmd(text)
    global imgs, centers
    *imgs, centers = ut.extraction(img, **params)
    update_axes()
    fig.canvas.draw_idle()
textbox.on_submit(on_cmd_submit)

if len(centers.shape) > 1:
    ut.add_toggleable_circles(fig, axs, np.roll(centers, 1, axis=1), 'v')

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