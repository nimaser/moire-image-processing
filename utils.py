# @author Nikhil Maserang
# @date 2023/04/13

import math
import numpy as np
import scipy.fft as spfft
import matplotlib.pyplot as plt
import sxm_reader as sxm
import cv2

def get_image_data(fname : str, display : bool = False, print_channels : bool = False) -> np.ndarray:
    """Grabs the data from the .sxm file as an ndarray."""
    file_object = sxm.NanonisSXM(fname)
    if print_channels: file_object.list_channels()
    image_data = file_object.retrieve_channel_data('Z')
    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(image_data, cmap="hot")
    return image_data

def subtract_mean(image_data : np.ndarray) -> np.ndarray:
    """Finds the mean of the flattened `image_data` array, then subtracts that value from each datapoint."""
    return image_data - np.mean(image_data)

def heatmap(image_data : np.ndarray, title : str = None) -> np.ndarray:
    """Plots the `image_data` on a heatmap using `np.imshow()`. Sets the plt.title to `title`."""
    plt.title(title)
    plt.imshow(image_data, cmap="hot")
    plt.colorbar()
    plt.show()
    #  extent=[0, image_data.shape[0], 0, image_data.shape[1]]

def run_shifted_fft(image_data : np.ndarray) -> np.ndarray:
    """Computes the DFT of `image_data` on outputs from -pi to pi."""
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

def bitmap_to_positions(bitmap : np.ndarray) -> list[tuple[int, int]]:
    """Extracts the positions of all 1s in a bitmap."""
    rows, cols = bitmap.shape
    pos = []
    for r in range(rows):
        for c in range(cols):
            if bitmap[r][c] == 1: pos.append((r, c))
    return pos

# https://stackoverflow.com/a/10847911
def order_vertex_positions(vertices : list[tuple[int, int]]) -> None:
    """Reorders a shuffled list of polygon vertices using a polar sweep method."""
    # compute the centroid
    x_vals, y_vals = zip(*vertices)
    x_sum, y_sum = sum(x_vals), sum(y_vals)
    centroid = [x_sum / len(vertices), y_sum / len(vertices)]

    # gets polar angle via arctan(delta_y / delta_x)
    polar_angle = lambda point : math.atan2(point[1] - centroid[1], point[0] - centroid[0])
    # sort by polar angle
    vertices.sort(key=polar_angle)

def get_hexagon_diagonals(vertices : list[tuple[int, int]]) -> list[tuple[tuple[int, int]]]:
    """Returns a list of three pairs of points, each pair representing the two vertices composing a diagonal. Assumes `vertices` are in order."""
    diag1 = (vertices[0], vertices[3])
    diag2 = (vertices[1], vertices[4])
    diag3 = (vertices[2], vertices[5])
    return [diag1, diag2, diag3]

def get_diagonals_length(diagonals: list[tuple[tuple[int, int]]]) -> list[int]:
    """Given `diagonals`, a list of diagonals, returns their lengths."""
    return [np.linalg.norm([pt2[1] - pt1[1], pt2[0] - pt1[0]]) for pt1, pt2 in diagonals]

# https://stackoverflow.com/a/76020025/10943551
def stretch_image(img : np.ndarray, P : tuple[int, int], V : tuple[int, int], S : float):
    """Stretches `img` along the direction `V` by a scale factor `S` such that a line passing
    through `P` which is perpendicular to `V` is not modified."""
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