# @author Nikhil Maserang
# @date 2023/08/14

import cv2
from enum import Enum
import sxm_reader as sxm
import numpy as np
import scipy.fft as spfft
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.backend_bases import KeyEvent

### MISC ###

QUIVER_PROPS = dict(angles='xy', scale_units='xy', scale=1)

def get_sxm_data(fname : str, print_channels : bool = False) -> np.ndarray:
    """Grabs the data from the .sxm file as an ndarray."""
    file_object = sxm.NanonisSXM(fname)
    if print_channels: file_object.list_channels()
    image_data = file_object.retrieve_channel_data('Z')
    return image_data

def centroid2D(vertices : np.ndarray) -> np.ndarray:
    """Finds the centroid of a set of xy points."""
    length = vertices.shape[0]
    xvals, yvals = vertices.T
    return np.array((np.sum(xvals) / length, np.sum(yvals) / length))

def rotate2D(v : np.ndarray, theta : float) -> np.ndarray:
    """Rotates vector `v` counterclockwise by `theta` radians."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ v.T

def get_transformation_matrix2D(u1 : np.ndarray, u2 : np.ndarray, v1 : np.ndarray, v2 : np.ndarray) -> np.ndarray:
    """Solves for the 2D transformation matrix which takes `u1` to `v1` and `u2` to `v2`."""
    M = np.linalg.inv(np.vstack((u1, u2)))
    y1, y2 = np.squeeze(np.vsplit(np.vstack((v1, v2)).T, 2))
    x1 = M @ y1.T
    x2 = M @ y2.T
    return np.vstack((x1, x2))

# https://stackoverflow.com/a/10847911
def order_vertices(vertices : np.ndarray) -> np.ndarray:
    """Orders a shuffled list of polygon vertices using a polar sweep. Not in place."""  
    # get x and y components of radial vectors between centroid and vertices;
    # effectively centers the vertices around (0, 0)
    dx, dy = (vertices - centroid2D(vertices)).T
    
    # compute polar angle via arctan(delta_y / delta_x), then sort
    angles = np.arctan2(dy, dx)
    indices = np.argsort(angles)
    
    # reorder the vertices and return
    return vertices[indices].copy()

def add_toggleable_circles(fig : Figure, axs : list[Axes], points : np.ndarray, key : str) -> None:
    """Adds circles for each point in `points` to each axis in `axs`, adding a visibility toggle."""
    circleslist = []
    for ax in axs.flatten():
        circles = []
        for x, y in points:
            circles.append(Circle((x, y), 1))
        circles = PatchCollection(circles, color='r', alpha=0.5)
        ax.add_collection(circles)
        circleslist.append(circles)

    def toggle_visibility(event : KeyEvent):
        if event.key == key:
            for circles in circleslist:
                circles.set_visible(not circles.get_visible())
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", toggle_visibility)
    
def add_processing_sequence(fig : Figure, ax : Axes, usecb : bool, *imgs : np.ndarray) -> None:
    """Displays several images in sequence, using , and . to scroll between them."""
    index = 0
    im = ax.imshow(imgs[index], cmap="gray")
    if usecb: cb = plt.colorbar(im, ax=ax, location="bottom")
    
    def change_image(event : KeyEvent):
        nonlocal index
        if   event.key == ',': shift = -1
        elif event.key == '<': shift = -2
        elif event.key == '.': shift = 1
        elif event.key == '>': shift = 2
        else: return
        index  = (index + shift) % len(imgs)
        
        im = ax.imshow(imgs[index], cmap="gray")
        if usecb: cb.update_normal(im)
        fig.canvas.draw_idle()
        
    fig.canvas.mpl_connect("key_press_event", change_image)

### IMAGES ###

def run_shifted_fft(image_data : np.ndarray) -> np.ndarray:
    """Computes the FFT of `image_data` on outputs from -pi to pi."""
    return spfft.fftshift(spfft.fft2(image_data))

def set_overlay_value(img : np.ndarray, pos : np.ndarray, size : int, val) -> None:
    """Sets `img` values to `val` in a rectangle of sidelengths 2*`size` about `pos`, in place."""
    b0, b1 = pos - size
    length = 2*size + 1
    img[b0:b0+length, b1:b1+length] = val

def get_circular_kernel(size : int) -> np.ndarray:
    """Returns a square array with sidelength `size` containing a circular arrangement of 1s."""
    arr = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            p = np.array([r, c]) + np.array([0.5, 0.5])
            v = (np.array(arr.shape) / 2) - p
            d = np.linalg.norm(v)
            if d < (size / 2):
                arr[r, c] = 1
    return arr

def get_border_mask(shape : np.ndarray, width : int) -> np.ndarray:
    """Returns a binary mask for the border of an img with given `shape`. Interior has 1s, border has 0s."""
    border = np.zeros(shape)
    inset = np.ones(shape - width*2)
    border[width:shape[0] - width, width:shape[1] - width] = inset
    return border

def scale_to_uint8(data : np.ndarray, expand : bool = True, convert : bool = True) -> np.ndarray:
    """Scales an array to fill the uint8 value range (as completely as possible if `expand` is True.)."""
    # shift data so that it's all nonnegative and has a minimum value of 0
    shifted = data - np.min(data)
    
    if expand or np.max(shifted) >= 256:
        # scale the data so the max is 255
        scaled = shifted * (255 / np.max(shifted))
    else: 
        # if the maximum value was already within the range and no expanding, we're done
        scaled = shifted

    return scaled.astype(np.uint8) if convert else scaled

def fill_holes_binary(binary : np.ndarray, initial : np.ndarray) -> np.ndarray:
    """Fills holes in `binary` starting at the specifed `initial` point."""
    assert binary.dtype == np.uint8, "binary array must be of dtype np.uint8"
    
    # apply a floodfill to the image to fill the outside of the shapes we want to fill in
    holes = binary.copy()
    shape = np.array(binary.shape)
    mask = np.zeros(shape + np.array((2, 2)), dtype=np.uint8)
    
    cv2.floodFill(holes, mask, initial, 1)
    
    # invert the filled image to get the negative of the original binary image (the holes)
    holes = np.logical_not(holes)
    
    # logical or combines the original image with the holes
    filled = np.logical_or(binary, holes).astype(np.uint8)
    return filled
    
def clean_edges_binary(binary : np.ndarray) -> np.ndarray:
    """Removes any blobs attached to the edge which have value 1 by flood filling with value 0."""
    assert binary.dtype == np.uint8, "binary array must be of dtype np.uint8"
    
    # repeatedly floodfill from each point on the border of the binary image
    cleaned = binary.copy()
    shape = np.array(binary.shape)
    mask = np.zeros(shape + np.array((2, 2)), dtype=np.uint8)
    
    borderindices = np.argwhere(0 == get_border_mask(shape, 1))
    for borderpoint in borderindices:
        # flip so rows are y and cols are x
        cv2.floodFill(cleaned, mask, borderpoint[::-1], 0)

    return cleaned
    
def get_blob_centroids(img : np.ndarray) -> np.ndarray:
    """Returns a list of indices of blob centroids."""
    centroids = []
    
    # find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # X is column, Y is row, so swap
            centroids.append((cY, cX))
    
    return np.array(centroids)

# automatic, manual, and adaptive thresholding options
class THRESH_MODE(Enum):
    AUTO = 1
    MANUAL = 2
    ADAPTIVE = 3

# process data using various parameters
def process(data : np.ndarray,
            flip : bool,
            mode : THRESH_MODE,
            trsh : int,
            blck : int,
            thrc : int,
            init : np.ndarray,
            smsz : int):
    """
    Carries out the processing steps necessary to extract peaks from an STM moire image.
    - `data` is the original np.uint8 array
    - `flip` is whether to invert it
    - `mode` is THRESH_MODE.AUTO, THRESH_MODE.MANUAL, or THRESH_MODE.ADAPTIVE
    - `trsh` is the threshold value for manual thresholding
    - `blck` is the blocksize for adaptive thresholding
    - `thrc` is the C value for adaptive thresholding
    - `init` is the initial point to floodfill from when filling holes
    - `smth` is the smoothing kernel size used when smoothing the edges of blobs 
    
    Returns a list of
    - `data`        the original data
    - inverted      inverted data (if flip == True)
    - `binary`      binarized data
    - `filled`      binarized data with filled in holes
    - `smoothed`    filled in data after edges have been smoothed
    - `cleaned`     smoothed data after any blobs touching edges have been removed
    - `centers`     xy coordinates of centers of blobs
    """
    assert data.dtype is np.uint8, "input data must be of dtype np.uint8"
    
    inverted = np.logical_xor(data).astype(np.uint8) if flip else data
    
    if mode == THRESH_MODE.AUTO:
        _, binary = cv2.threshold(inverted, trsh, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mode == THRESH_MODE.MANUAL:
        _, binary = cv2.threshold(inverted, trsh, 1, cv2.THRESH_BINARY)
    if mode == THRESH_MODE.ADAPTIVE:
        binary = cv2.adaptiveThreshold(inverted, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blck, thrc)

    filled = fill_holes_binary(binary, init)

    smoothed = spnd.median_filter(filled, footprint=ut.get_circular_kernel(smsz))

    cleaned = clean_edges_binary(smoothed)

    centers = get_blob_centroids(cleaned)
    
    return data, inverted, binary, filled, smoothed, cleaned, centers
    
### HEXAGONS ###

def plot_hex_radii(ax : Axes, vertices):
    """Plots the radii of a hexagon (vectors from its centroid to its vertices)."""
    centroid = centroid2D(vertices)
    radii = vertices - centroid
    
    # need 6 copies of centroid, one per vector
    x, y = np.resize(centroid, (6, 2)).T
    
    u, v = radii.T
    colors=['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
    ax.quiver(x, y, u, v, color=colors, **QUIVER_PROPS)

def verify_hexagon_shape(vertices : np.ndarray, tolerance : float, bgd : np.ndarray = None) -> np.ndarray:
    """Given an ordered list of hexagon vertices, returns whether the sum of the radial
    vectors on either side of a vertex sum to the radial vector of that vertex. If `bgd`
    is provided, a visualization is plotted over the background."""
    # get radial vectors
    vectors = vertices - centroid2D(vertices)
    
    # roll vertices ahead and behind by 1, then sum
    ahead = np.roll(vectors, 1, axis=0)
    behind = np.roll(vectors, -1, axis=0)
    vector_sum = ahead + behind
    
    if bgd is not None:
        # create figure to show summations
        fig, axs = plt.subplots(3, 2, figsize=(20, 30))
        for ax in axs.flatten():
            ax.imshow(bgd, cmap="hot")

        # stack the sets of vectors so each individual axis 0 array contains corresponding vectors,
        # then reshape to match shape of axs
        V = np.stack([vectors, ahead, behind, vector_sum], axis=1).reshape(3, 2, 4, 2)

        # vector origin should be centroid; one copy needed per vector
        x, y = np.resize(centroid2D(vertices), (4, 2)).T

        for r in range(3):
            for c in range(2):
                # V[r, c] contains the four vectors which need to be plotted
                axs[r, c].quiver(x, y, *V[r, c].T, color=["red", "green", "green", "blue"], **QUIVER_PROPS)
    
    # compare using an absolute tolerance
    return np.all(np.isclose(vectors, vector_sum, rtol=0, atol=tolerance))
    
def select_invariant_vector(vertices : np.ndarray) -> int:
    """Returns the index of the vector that should be invariant after the transformation."""
    # choose invariant vector as the first one in the list;
    # not sure if there are reasons to choose a different vector
    return 0

def get_correction_matrix(vertices : np.ndarray, required_length : int) -> np.ndarray:
    """
    Finds the transformation matrix required to rectify the distorted hexagon given by `vertices`,
    an ordered list of points, both by stretching and scaling to the proper dimensions.
    """
    # get invariant vector's index
    invariant_index = select_invariant_vector(vertices)
    
    # get initial vectors
    vectors = vertices - centroid2D(vertices)
    u1 = vectors[invariant_index]
    u2 = vectors[invariant_index + 2]

    # get final vectors by scaling and then rotating that scaled one
    v1 = u1 / np.linalg.norm(u1) * required_length
    v2 = rotate2D(v1, 2 * np.pi / 3)
    
    # O moves the origin to centroid, -O moves it back
    O = np.eye(3)
    O[0:2, 2] = -centroid2D(vertices)
    
    # this is the 2D transformation matrix that performs the stretch, but it's not centered at the origin
    M = get_transformation_matrix2D(u1, u2, v1, v2)
    
    # embed M inside of a 3x3 matrix T
    T = np.eye(3)
    T[0:2, 0:2] = M
    
    # O moves hexagon to origin, M applies the
    # transformation to stretch the hexagon,
    # then -O moves the new hexagon back from
    # the origin; only the first two rows are
    # needed in affine transforms
    return (-O @ T @ O)[0:2, :]

def transform_image(data : np.ndarray, A : np.ndarray) -> np.ndarray:
    """
    Transforms `data` according to the affine transformation matrix `A`.
    """
    return cv2.warpAffine(data, A, tuple(data.shape))
    
    