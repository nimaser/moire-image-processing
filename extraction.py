# @author Nikhil Maserang
# @date 2023/09/02

import numpy as np

from tkinter.simpledialog import askstring

import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.backend_bases import KeyEvent

import utils as ut

###########################################################

# 002, 004, 005, 006, 008

# get image data
fname = "images/m002.sxm"
if fname.endswith(".sxm"):
    img = ut.get_sxm_data(fname, False)
    img = ut.scale_to_uint8(img)
else: exit()
    
###########################################################

# set default extraction parameters
params = dict(
    flip = 1,
    mode = "adaptive",
    trsh = 112,
    blck = 15,
    thrc = 2,
    init = (0,0),
    smth = 7
)
def gen_params_str():
    out = ""
    out += f"-flip: {params['flip']}\n"
    out += f"-mode: {params['mode']}\n"
    out += f"-trsh: {params['trsh']}\n"
    out += f"-blck: {params['blck']}\n"
    out += f"-thrc: {params['thrc']}\n"
    out += f"-init: {params['init']}\n"
    out += f"-smth: {params['smth']}\n"
    out += f"-expu\n"
    out += f"-expw"
    return out

# command handling
cmdhandler = ut.CommandProcessor()
cmdhandler.add_cmd("-flip", 1, lambda x: params.update(dict(flip=int(x))))
cmdhandler.add_cmd("-mode", 1, lambda x: params.update(dict(mode=x)))
cmdhandler.add_cmd("-trsh", 1, lambda x: params.update(dict(trsh=int(x))))
cmdhandler.add_cmd("-blck", 1, lambda x: params.update(dict(blck=int(x))))
cmdhandler.add_cmd("-thrc", 1, lambda x: params.update(dict(thrc=int(x))))
cmdhandler.add_cmd("-smth", 1, lambda x: params.update(dict(smth=int(x))))
cmdhandler.add_cmd("-expu", 0, lambda: print("u"))
cmdhandler.add_cmd("-expw", 0, lambda: print("w"))

def parse_init_tuple(x, y):
    try:
        x, y = int(x), int(y)
        params.update(dict(init=(x, y)))
    except: pass
cmdhandler.add_cmd("-init", 2, parse_init_tuple)

###########################################################

# create figure
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
fig.suptitle("1 toggles unweighted centroids; 2 toggles weighted centroids;\nw opens overlay window; - opens command window")
mc = MultiCursor(None, axs.flatten(), horizOn=True, color='b', lw=1)

# axes titles
titles = [
    "original",
    "binarized",
    "filled in",
    "blobs smoothed",
    "edges cleaned",
    "masked"
]
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

# command processing
def open_command_window(event : KeyEvent) -> None:
    if event.key == '-':
        # parse and set new params
        cmd = askstring("CmdPromptWindow", gen_params_str(), initialvalue="-")
        cmdhandler.process_cmd(cmd)
        
        # recalculate imgs and centers
        global imgs, uw_centers, w_centers
        *imgs, uw_centers, w_centers = ut.extraction(img, **params)
        
        # replot imgs and centers and redraw
        global uw_cid, w_cid, uw_circleslist, w_circleslist
        update_axes()
        ut.remove_toggleable_circles(fig, uw_cid, uw_circleslist)
        ut.remove_toggleable_circles(fig, w_cid, w_circleslist)
        uw_cid = None
        w_cid = None
        uw_circleslist = []
        w_circleslist = []
        if len(uw_centers.shape) > 1:
            uw_cid, uw_circleslist = ut.add_toggleable_circles(fig, axs, np.roll(uw_centers, 1, axis=1), 'r', '1')
        if len(w_centers.shape) > 1:
            w_cid, w_circleslist = ut.add_toggleable_circles(fig, axs, np.roll(w_centers, 1, axis=1), 'b', '2')
        fig.canvas.draw()
fig.canvas.mpl_connect("key_press_event", open_command_window)

# overlay window
def open_overlay_window(event : KeyEvent):
    if event.key == 'w':
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle("< , . > to move through image queue")
        ut.add_image_sequence(fig, ax, True, imgs, titles)
        if len(uw_centers.shape) > 1:
            ut.add_toggleable_circles(fig, np.array([ax]), np.roll(uw_centers, 1, axis=1), 'r', '1')
        if len(w_centers.shape) > 1:
            ut.add_toggleable_circles(fig, np.array([ax]), np.roll(w_centers, 1, axis=1), 'b', '2')
        fig.show()
fig.canvas.mpl_connect("key_press_event", open_overlay_window)

###########################################################

# calculate initial data and display on axes
*imgs, uw_centers, w_centers = ut.extraction(img, **params)
update_axes()

# add circles if any were found
uw_cid = None
w_cid = None
uw_circleslist = []
w_circleslist = []
if len(uw_centers.shape) > 1:
    uw_cid, uw_circleslist = ut.add_toggleable_circles(fig, axs, np.roll(uw_centers, 1, axis=1), 'r', '1')
if len(w_centers.shape) > 1:
    w_cid, w_circleslist = ut.add_toggleable_circles(fig, axs, np.roll(w_centers, 1, axis=1), 'b', '2')
      
plt.show()

###########################################################

points = [ut.Point(center) for center in w_centers]
if len(points) < 3: exit()

fractional_tolerance = 0.4

pls0 = ut.ParallelLineSet(points, fractional_tolerance)
pls1 = ut.ParallelLineSet(pls0  , fractional_tolerance)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(imgs[0], cmap="gray")
ut.add_toggleable_circles(fig, np.array([ax]), np.roll(w_centers, 1, axis=1), 'b', '1')

for line, color, key in zip(pls0.lines, ['r', 'g', 'c'], ['3', '3', '3']):
    ut.add_toggleable_circles(fig, np.array([ax]), np.roll(line.get_points_as_ndarray(), 1, axis=1), color, key)
    pos, vec = line.get_spanning_vector()
    ut.add_toggleable_vectors(fig, np.array([ax]), np.roll(pos, 1, axis=0), np.roll(vec, 1, axis=0), color, key)

for line, color, key in zip(pls1.lines, ['r', 'g', 'c', 'y'], ['4', '4', '4', '4']):
    ut.add_toggleable_circles(fig, np.array([ax]), np.roll(line.get_points_as_ndarray(), 1, axis=1), color, key)
    pos, vec = line.get_spanning_vector()
    ut.add_toggleable_vectors(fig, np.array([ax]), np.roll(pos, 1, axis=0), np.roll(vec, 1, axis=0), color, key)
    
latvec0 = pls0.get_lattice_vector()
latvec1 = pls1.get_lattice_vector()
vec = np.concatenate([latvec0[:, np.newaxis], latvec1[:, np.newaxis]], axis=1)

latpos = pls0.get_origin()
pos = np.concatenate([latpos[:, np.newaxis], latpos[:, np.newaxis]], axis=1)

ut.add_toggleable_vectors(fig, np.array([ax]), np.roll(pos, 1, 0), np.roll(vec, 1, 0), 'k', '2')
    
plt.show()