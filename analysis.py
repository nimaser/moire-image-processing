# @author Nikhil Maserang
# @date 2023/09/02

import numpy as np

import matplotlib.pyplot as plt

import utils as ut

fname = "images/m004.sxm"
if fname.endswith(".sxm"):
    img = ut.get_sxm_data(fname, False)
    img = ut.scale_to_uint8(img)
    
w_centers = np.array([[108.28071992, 115.32948718],
                      [106.29380227,  84.34548574],
                      [103.73518844,  52.87520067],
                      [101.68850464,  21.43496939],
                      [ 79.07570555,  98.42257767],
                      [ 76.76898488,  67.14137015],
                      [ 74.38563379,  35.7108429 ],
                      [ 51.44463984, 112.73160431],
                      [ 49.01109293,  81.38893658],
                      [ 46.63114765,  49.99195868],
                      [ 44.17495357,  18.34213895],
                      [ 20.54954306,  95.58630693]])

fractional_tolerance = 0.4

points = [ut.Point(center) for center in w_centers]

pls0 = ut.ParallelLineSet(points, fractional_tolerance)
pls1 = ut.ParallelLineSet(pls0  , fractional_tolerance)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray")
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