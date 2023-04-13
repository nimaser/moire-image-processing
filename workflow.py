# @author Nikhil Maserang
# @date 2023/04/13

import numpy as np
import scipy.fft as spfft
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
import sxm_reader as sxm

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

def heatmap(image_data : np.ndarray, size : int = 10) -> np.ndarray:
    """Plots the `image_data` on a heatmap using `np.imshow()`. `size` is the matplotlib figure size."""
    plt.figure(figsize=(size, size))
    plt.imshow(image_data, cmap="hot")

def run_shifted_fft(image_data : np.ndarray) -> np.ndarray:
    """Computes the DFT of `image_data` on outputs from -pi to pi."""
    return spfft.fftshift(spfft.fft2(image_data))

def plot_transformation(image_data : np.ndarray, transformation : np.ndarray) -> np.ndarray:
    """Applies the given `transformation` matrix to the `image_data`."""
    output = spnd.affine_transform(image_data, transformation)
    output_fft = run_shifted_fft(output)
    heatmap(output)
    heatmap(abs(output_fft))