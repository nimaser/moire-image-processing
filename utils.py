# @author Nikhil Maserang
# @date 2023/04/13

import math
import numpy as np
import scipy.fft as spfft
import matplotlib.pyplot as plt
import sxm_reader as sxm
import cv2

def get_image_data(fname : str, print_channels : bool = False) -> np.ndarray:
    """Grabs the data from the .sxm file as an ndarray."""
    file_object = sxm.NanonisSXM(fname)
    if print_channels: file_object.list_channels()
    image_data = file_object.retrieve_channel_data('Z')
    return image_data

def run_shifted_fft(image_data : np.ndarray) -> np.ndarray:
    """Computes the FFT of `image_data` on outputs from -pi to pi."""
    return spfft.fftshift(spfft.fft2(image_data))

# https://stackoverflow.com/a/33792838
def superimpose_center(canvas : np.ndarray, image : np.ndarray) -> None:
    """Superimposes the `image` array onto the center of the `canvas` array, in-place."""

    def get_slices_ND(canvas_shape : tuple, image_shape : tuple) -> tuple:
        """Gets the slices needed for all axes."""

        def get_slices_1D(canvas_axis_length : int, image_axis_length : int) -> tuple[slice, slice]:
            """Helper function to return the slice needed along one particular axis."""
            # | buffer | image_axis_length | buffer |
            buffer = abs(canvas_axis_length - image_axis_length) // 2

            if canvas_axis_length > image_axis_length:
                # return slice of indexes which need to be overwritten, then slice(None)??
                return slice(buffer, buffer + image_axis_length), slice(None)
            else:
                # return slice(None), then some mysterious slice which I don't understand
                return slice(None), slice(buffer, buffer + canvas_axis_length)
        
        # pair up corresponding axis lengths of canvas and image
        pairs = zip(canvas_shape, image_shape)

        return zip(*(get_slices_1D(*pair) for pair in pairs))
    
    # black magic
    canvas_indexes, image_indexes = get_slices_ND(canvas.shape, image.shape)
    canvas[canvas_indexes] = image[image_indexes]

def blot_center(img : np.ndarray, blot_size : int) -> None:
    """Blots out a square of sidelength `blot_size` from the center of `img`, in-place."""
    blot = np.zeros((blot_size, blot_size))
    superimpose_center(img, blot)
 
def plot_vector(ax, O : np.ndarray, v : np.ndarray, c : str = 'b'):
    """Plots a vector on an Axes, returning the resulting Line2D."""
    return ax.quiver(*O, *v, color=c, angles='xy', scale_units='xy', scale=1)
    
def centroid2D(vertices : np.ndarray) -> np.ndarray:
    """Finds the centroid of a set of xy points."""
    length = vertices.shape[0]
    xvals, yvals = vertices.T
    return np.array((np.sum(xvals) / length, np.sum(yvals) / length))

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
    return vertices[indices]

# https://stackoverflow.com/a/76020025/10943551
def stretch_image(img : np.ndarray, P : tuple[int, int], V : tuple[int, int], S : float):
    """Stretches `img` (convered to np.dtype.int16) along the direction `V` by a scale factor `S`
    such that a line passing through `P` which is perpendicular to `V` is not modified."""
    img = img.astype(np.int16)
    M_translate_forward = np.array([[1, 0, -P[0]],
                                    [0, 1, -P[1]],
                                    [0, 0, 1   ]], dtype = float)
    
    Vn = V / np.linalg.norm(V)
    cos_phi = Vn[1]
    sin_phi = Vn[0]
    
    M_rotate_forward = np.array([[cos_phi, -sin_phi, 0], 
                                 [sin_phi,  cos_phi, 0], 
                                 [      0,        0, 1]])
    
    scale_W = 1
    scale_H = S
    M_scale = np.array([[scale_W,       0, 0], 
                        [      0, scale_H, 0], 
                        [      0,       0, 1]])
    
    M_rotate_backward = np.linalg.inv(M_rotate_forward)
    M_translate_backward = np.linalg.inv(M_translate_forward)
    
    M = M_translate_backward @ M_rotate_backward @ M_scale@ M_rotate_forward @ M_translate_forward
    
    return cv2.warpAffine(img, M[:2], img.shape[:2][::-1], img.shape)
    
def verify_hexagon_shape(vertices : np.ndarray, tolerance : float) -> np.ndarray:
    """Given an ordered list of hexagon vertices, returns whether the sum of the radial
    vectors on either side of a vertex sum to the radial vector of that vertex."""
    # get radial vectors
    vectors = vertices - centroid2D(vertices)
    
    # roll vertices ahead and behind by 1, then sum
    ahead = np.roll(vectors, 1, axis=0)
    behind = np.roll(vectors, -1, axis=0)
    vector_sum = ahead + behind
    
    # compare using an absolute tolerance
    return np.all(np.isclose(vectors, vector_sum, rtol=0, atol=tolerance))

def rotate2D(v : np.ndarray, theta : float) -> np.ndarray:
    """Rotates `v` counterclockwise by `theta` radians."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ v.T
    
def get_transformation_matrix2D(u1 : np.ndarray, u2 : np.ndarray, v1 : np.ndarray, v2 : np.ndarray) -> np.ndarray:
    """Solves for the 2D transformation matrix which takes `u1` to `v1` and `u2` to `v2`."""
    M = np.linalg.inv(np.vstack((u1, u2)))
    y1, y2 = np.squeeze(np.vsplit(np.vstack((v1, v2)).T, 2))
    x1 = M @ y1.T
    x2 = M @ y2.T
    return np.vstack((x1, x2))
    
def select_invariant_vector(vertices : np.ndarray) -> int:
    """Returns the index of the vector that should be invariant after the transformation."""
    # choose invariant vector as the first one in the list;
    # not sure if there are reasons to choose a different vector
    return 0

def get_image_transform(vertices : np.ndarray, required_length : int) -> np.ndarray:
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
    
    # this is the 2D transformation matrix
    return get_transformation_matrix2D(u1, u2, v1, v2)

def transform_image(data : np.ndarray, vertices : np.ndarray, required_length : int) -> np.ndarray:
    """
    Transforms the image to rectify the distorted hexagon.
    """
    # this is the 2D transformation matrix
    M = get_image_transform(vertices, required_length)

    # O moves the origin to centroid, -O moves it back
    O = np.eye(3)
    O[0:2, 2] = -centroid2D(vertices)
    
    # embed M inside of a 3x3 matrix T
    T = np.eye(3)
    T[0:2, 0:2] = M

    # affine tranform A moves hexagon to origin,
    # applies transformation to stretch hexagon,
    # then moves new hexagon back from origin;
    # only the first two rows are relevant for
    # affine transforms
    A = (-O @ T @ O)[0:2, :]
    
    return cv2.warpAffine(data, A, tuple(data.shape))
    
    