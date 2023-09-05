# @author Nikhil Maserang
# @date 2023/09/02

import functools
from collections.abc import Callable
from typing import Self

import sxm_reader as sxm

import cv2
import numpy as np
import scipy.fft as spfft
import scipy.ndimage as spnd

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.backend_bases import KeyEvent
from matplotlib.artist import Artist
from matplotlib.quiver import Quiver

########################################################### MATPLOTLIB

def add_visibility_toggle(fig : Figure, artists : list[Artist], key : str) -> int:
    """Adds a callback to toggle the visibility of the provided `artist` when `key` is pressed."""
    def toggle_visibility(event : KeyEvent):
        if event.key == key:
            for artist in artists:
                artist.set_visible(not artist.get_visible())
            fig.canvas.draw_idle()
    cid = fig.canvas.mpl_connect("key_press_event", toggle_visibility)

def add_toggleable_circles(fig : Figure, axs : np.ndarray[Axes], points : np.ndarray, color : str, key : str) -> tuple[int, list[PatchCollection]]:
    """Adds circles for each point in `points` to each axis in `axs`, adding a visibility toggle."""
    circleslist = []
    for ax in axs.flatten():
        circles = []
        for x, y in points:
            circles.append(Circle((x, y), 1))
        circles = PatchCollection(circles, color=color, alpha=0.5)
        ax.add_collection(circles)
        circleslist.append(circles)
    cid = add_visibility_toggle(fig, circleslist, key)
    return cid, circleslist

def add_toggleable_vectors(fig : Figure, axs : np.ndarray[Axes], positions : np.ndarray, vectors : np.ndarray, color : str, key : str) -> tuple[int, list[Quiver]]:
    """Adds vectors for each point and vector in `positions` and `vectors` to each axis in `axs`, adding a visibility toggle."""
    quiverlist = []
    for ax in axs.flatten():
        quiverlist.append(ax.quiver(*positions, *vectors, color=color, angles='xy', scale_units='xy', scale=1))
    cid = add_visibility_toggle(fig, quiverlist, key)
    return cid, quiverlist

def remove_toggleable_circles(fig : Figure, cid, circleslist : list[PatchCollection]) -> None:
    """Removes the circles added by `add_toggleable_circles`."""
    for circles in circleslist: circles.remove()
    fig.canvas.mpl_disconnect(cid)
    
def remove_toggleable_vectors(fig : Figure, cid, quiverlist : list[Quiver]) -> None:
    """Removes the vectors added by `add_toggleable_vectors`."""
    for quiver in quiverlist: quiver.remove()
    fig.canvas.mpl_disconnect(cid)

def add_artist_sequence(fig : Figure, ax : Axes, artists : list[Artist], title : str) -> None:
    """Displays several artists in sequence, using < , . and > to scroll between them."""
    index = 0
    ax.set_title(title + "\n" + "▢"*index + "▣" + "▢"*(len(artists) - index - 1))
    for artist in artists: artist.set_visible(False)
    artists[index].set_visible(True)
    
    def change_artist(event : KeyEvent):
        nonlocal index
        if   event.key == ',': shift = -1
        elif event.key == '<': shift = -2
        elif event.key == '.': shift = 1
        elif event.key == '>': shift = 2
        else: return
        index  = (index + shift) % len(artists)
        
        ax.set_title(title + "\n" + "▢"*index + "▣" + "▢"*(len(artists) - index - 1))
        for artist in artists: artist.set_visible(False)
        artists[index].set_visible(True)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", change_artist)
    
def add_image_sequence(fig : Figure, ax : Axes, usecb : bool, imgs : np.ndarray, titles : list[str]) -> None:
    """Displays several images in sequence, using < , . and > to scroll between them."""
    index = 0
    ax.set_title(titles[index] + "\n" + "▢"*index + "▣" + "▢"*(len(imgs) - index - 1))
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
        
        ax.set_title(titles[index] + "\n" + "▢"*index + "▣" + "▢"*(len(imgs) - index - 1))
        im = ax.imshow(imgs[index], cmap="gray")
        if usecb: cb.update_normal(im)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", change_image)

########################################################### MISC

def get_sxm_data(fname : str, print_channels : bool = False) -> np.ndarray:
    """Grabs the data from the .sxm file as an ndarray."""
    file_object = sxm.NanonisSXM(fname)
    if print_channels: file_object.list_channels()
    image_data = file_object.retrieve_channel_data('Z')
    return image_data

class CommandProcessor:
    def __init__(self):
        """Utility class to """
        self.charbuffer = []
        
        self.flags = []
        self.num_args = []
        self.functions = []
        
    def add_cmd(self, flag : str, num_args : int, func : Callable) -> None:
        if not flag.startswith("-"): raise Exception("flags must start with a dash (-) character")
        if flag not in self.flags:
            self.flags.append(flag)
            self.num_args.append(num_args)
            self.functions.append(func)
            
    def process_cmd(self, cmdstring) -> None:
        if cmdstring is None or len(cmdstring) == 0: return
        cmds = []
        
        words = "".join(cmdstring).split()
        cmd = []
        while len(words) > 0:
            if words[0].startswith("-"):
                if len(cmd) > 0: cmds.append(cmd)
                cmd = [words.pop(0)]
            else:
                cmd.append(words.pop(0))
        if len(cmd) > 0: cmds.append(cmd)
        
        func_queue = []
        for cmd in cmds:
            flag, args = cmd.pop(0), cmd
            
            if flag not in self.flags: return
            ind = self.flags.index(flag)
            if len(args) != self.num_args[ind]: return
            
            func_queue.append(functools.partial(self.functions[ind], *args))
        for func in func_queue: func()
        
    def append_char(self, char):
        if char == 'enter':
            self.process_cmd(self.charbuffer)
            self.charbuffer = []
        elif char == 'backspace':
            if len(self.charbuffer) > 0: self.charbuffer.pop()
        elif len(char) == 1:
            self.charbuffer.append(char)

########################################################### IMAGES

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
    
def get_uw_centroids(img : np.ndarray) -> np.ndarray:
    """Returns a list of indices of blob centroids, unweighted by intensity."""
    centroids = []
    
    # find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # X is column, Y is row, so swap
            centroids.append((cy, cx))
    
    return np.array(centroids)

# https://stackoverflow.com/a/65344493/22391526
def get_w_centroids(img : np.ndarray) -> np.ndarray:
    """Returns a list of indices of blob centroids, weighted by intensity."""
    centroids = []
    
    # create a meshgrid for coordinate calculation
    r,c = np.shape(img)
    r_ = np.linspace(0,r,r+1)
    c_ = np.linspace(0,c,c+1)
    X, Y = np.meshgrid(c_, r_, sparse=False, indexing='xy')
    
    # find contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        # Get the boundingbox
        x,y,w,h = cv2.boundingRect(contour)

        # calculate x,y coordinate of center
        # Get the corresponding roi for calculation
        weights = img[y:y+h,x:x+w]
        roi_grid_x = X[y:y+h,x:x+w]
        roi_grid_y = Y[y:y+h,x:x+w]
        
        # get the weighted sum
        weighted_x = weights * roi_grid_x
        weighted_y = weights * roi_grid_y
        
        if np.sum(weights) != 0:
            cx = np.sum(weighted_x) / np.sum(weights)
            cy = np.sum(weighted_y) / np.sum(weights)
        
            centroids.append((cy, cx))
        
    return np.array(centroids)
        
# performs the extraction pipeline and returns intermediate steps
def extraction(data : np.ndarray,
               flip : bool,
               mode : str,
               trsh : int,
               blck : int,
               thrc : int,
               init : np.ndarray,
               smth : int):
    """
    Carries out the processing steps necessary to extract peaks from an STM moire image.
    - `data` is the original np.uint8 array
    - `flip` is whether to invert it
    - `mode` is "auto", "manual", or "adaptive"
    - `trsh` is the threshold value for manual thresholding
    - `blck` is the blocksize for adaptive thresholding
    - `thrc` is the C value for adaptive thresholding
    - `init` is the initial point to floodfill from when filling holes
    - `smth` is the smoothing kernel size used when smoothing the edges of blobs 
    
    Returns a list of
    - `data`        the original data
    - `flipped`     inverted data (if flip == True)
    - `binary`      binarized data
    - `filled`      binarized data with filled in holes
    - `smoothed`    filled in data after edges have been smoothed
    - `cleaned`     smoothed data after any blobs touching edges have been removed
    - `centers`     xy coordinates of centers of blobs
    """
    assert data.dtype == "uint8", f"input data must be of dtype np.uint8; got type {data.dtype}"
    
    flipped = 255 - data if flip else data
    
    if mode == "auto":
        _, binary = cv2.threshold(flipped, trsh, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mode == "manual":
        _, binary = cv2.threshold(flipped, trsh, 1, cv2.THRESH_BINARY)
    if mode == "adaptive":
        binary = cv2.adaptiveThreshold(flipped, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blck, thrc)

    filled = fill_holes_binary(binary, init)

    smoothed = spnd.median_filter(filled, footprint=get_circular_kernel(smth))

    cleaned = clean_edges_binary(smoothed)
    
    masked = cleaned * flipped

    unweighted_centers = get_uw_centroids(cleaned)
    weighted_centers = get_w_centroids(masked)
    
    return flipped, binary, filled, smoothed, cleaned, masked, unweighted_centers, weighted_centers

########################################################### DATA

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
    
########################################################### SCHEDULED FOR REPROGRAMMING

def plot_hex_radii(ax : Axes, vertices):
    """Plots the radii of a hexagon (vectors from its centroid to its vertices)."""
    centroid = centroid2D(vertices)
    radii = vertices - centroid
    
    # need 6 copies of centroid, one per vector
    x, y = np.resize(centroid, (6, 2)).T
    
    u, v = radii.T
    colors=['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
    ax.quiver(x, y, u, v, color=colors, angles='xy', scale_units='xy', scale=1)

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
                axs[r, c].quiver(x, y, *V[r, c].T, color=["red", "green", "green", "blue"], angles='xy', scale_units='xy', scale=1)
    
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
    
    

########################################################### LATTICE VECTOR IDENTIFICATION

class Point:
    def __init__(self, coords : np.ndarray):
        self.coords = coords
        self.x = coords[0]
        self.y = coords[1]
        
    def distance_to(self, other : Self) -> float:
        return np.sqrt(np.sum((other.coords - self.coords) ** 2))
    
    def vector_to(self, other : Self) -> np.ndarray:
        return other.coords - self.coords
    
    def add_vector(self, vector : np.ndarray) -> Self:
        return Point(self.coords + vector)
        
    def order_by_nearest(self, others : np.ndarray[Self]) -> np.ndarray[Self]:
        distances = np.array([self.distance_to(other) for other in others])
        return others[np.argsort(distances)]
    
    def pop_closest(self, others : list[Self]) -> Self:
        closest = self.order_by_nearest(np.array(others))[0]
        others.remove(closest)
        return closest
        
    def pop_closest_within_range(self, others : list[Self], range : float) -> Self | None:
        distances = np.array([self.distance_to(other) for other in others])
        idx = np.argsort(distances)
        distances, points = distances[idx], np.array(others)[idx]
        
        in_range = distances <= range
        if not np.any(in_range): return None
        
        # https://stackoverflow.com/a/1044443/22391526
        idx_first_in_range = np.nonzero(in_range)[0][0]
        closest = points[idx_first_in_range]
        others.remove(closest)
        return closest
        
class Line:
    def __init__(self, initial_points : tuple[Point, Point], tolerance : float):
        self.points = initial_points
        self.endpoints = [initial_points[0], initial_points[1]]
        self.vector = self.endpoints[0].vector_to(self.endpoints[1])
        self.range = np.linalg.norm(self.vector) * tolerance
        
    def get_points_as_ndarray(self) -> np.ndarray:
        return np.array([point.coords for point in self.points])
        
    def __contains__(self, point : Point):
        return point in self.points
    
    def __len__(self):
        return len(self.points)
        
    def extend_endpoint(self, index : bool, points : list[Point]) -> bool:
        new_point_area = self.endpoints[index].add_vector( (1 if index else -1) * self.vector)
        new = new_point_area.pop_closest_within_range(points, self.range)
        
        if new is not None:
            if new in self.points: raise Exception("duplicated point somehow")
            self.points.append(new)
            self.endpoints[index] = new
            return True
        return False
        
    def extend(self, points : list[Point]) -> None:
        while self.extend_endpoint(0, points): pass
        while self.extend_endpoint(1, points): pass
        
    def get_spanning_vector(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.endpoints[0].coords
        vec = (self.endpoints[1].coords - self.endpoints[0].coords)
        return pos, vec
    
    def get_lattice_vector(self) -> np.ndarray:
        return (self.endpoints[1].coords - self.endpoints[0].coords) / (len(self.points) - 1)
        
class ParallelLineSet:
    def __init__(self, point_source : list[Point] | Self, tolerance : float):
        self.unincluded_points = []
        
        if type(point_source) is list:
            self.points = point_source
            
            origin = self.points.pop(0)
            closest = origin.pop_closest(self.points)
        else:
            self.points = []
            for line in point_source.lines[1:]: self.points.extend(line.points)
            self.points.extend(point_source.unincluded_points)
            
            origin = point_source.lines[0].points[0]
            closest = origin.pop_closest(self.points)
            
            self.points.extend(point_source.lines[0].points[1:])
            
        init = Line([origin, closest], tolerance)
        init.extend(self.points)
        self.vector = init.vector
        
        self.lines = [init]
        self.tolerance = tolerance
        self.range = np.linalg.norm(self.vector) * tolerance
        
        self.get_parallel()
            
    def get_parallel(self):
        while len(self.points) > 0:
            origin = self.points.pop(0)
            
            new_point_area = origin.add_vector(self.vector)
            new_point = new_point_area.pop_closest_within_range(self.points, self.range)
            
            if new_point is not None:
                new_line = Line([origin, new_point], self.tolerance)
            else:
                new_point_area = origin.add_vector(-self.vector)
                new_point = new_point_area.pop_closest_within_range(self.points, self.range)
                if new_point is not None:
                    new_line = Line([new_point, origin], self.tolerance)
                else:
                    self.unincluded_points.append(origin)
                    continue

            new_line.extend(self.points)
            self.lines.append(new_line)
    
    def get_lattice_vector(self) -> np.ndarray:
        vec = np.array([0, 0])
        for line in self.lines:
            vec = vec + line.get_lattice_vector()
        return vec / len(self.lines)
    
    def get_origin(self) -> np.ndarray:
        return self.lines[0].points[0].coords
        
###########################################################