# @author Nikhil Maserang
# @date 2023/09/02

import numpy as np

from typing import Self

import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.backend_bases import KeyEvent

import utils as ut

###########################################################

class Point:
    def __init__(self, coords : np.ndarray):
        self.coords = coords
        self.x = coords[0]
        self.y = coords[1]
        
    def copy(self) -> Self:
        return Point(self.coords.copy())
        
    def distance_to(self, other : Self) -> float:
        return np.sqrt(np.sum((other.coords - self.coords) ** 2))
    
    def vector_to(self, other : Self) -> np.ndarray:
        return other.coords - self.coords
    
    def add_vector(self, vector : np.ndarray) -> Self:
        return Point(self.coords + vector)
        
    def order_by_nearest(self, others : np.ndarray[Self]) -> np.ndarray[Self]:
        distances = np.array([self.distance_to(other) for other in others])
        return others[np.argsort(distances)]
    
    def find_closest(self, others : np.ndarray[Self]) -> Self:
        return self.order_by_nearest(others)[0]
        
    def find_closest_within_range(self, others : np.ndarray[Self], range : float) -> Self | None:
        distances = np.array([self.distance_to(other) for other in others])
        idx = np.argsort(distances)
        distances, points = distances[idx], others[idx]
        
        in_range = distances <= range
        if not np.any(in_range): return None
        
        # https://stackoverflow.com/a/1044443/22391526
        idx_first_in_range = np.nonzero(in_range)[0][0]
        return points[idx_first_in_range]
        
class Line:
    def __init__(self, initials : list[Point], vector : np.ndarray, tolerance : float):
        """`initials[0]` + `vector` should equal `initials[1]`"""
        self.points = initials
        self.vector = vector
        self.tolerance = np.linalg.norm(vector) * tolerance
        
        self.endpoints = [initials[0], initials[1]]
        
    def get_points_as_ndarray(self) -> np.ndarray:
        return np.array([point.coords for point in self.points])
        
    def __contains__(self, point : Point):
        return point in self.points
        
    def extend_endpoint(self, index : bool, points : list[Point]) -> bool:
        new_point_area = self.endpoints[index].add_vector( (1 if index else -1) * self.vector)
        new = new_point_area.find_closest_within_range(points, self.tolerance)
        
        if new is not None:
            if new in self.points: raise Exception("duplicated point somehow")
            self.points.append(new)
            self.endpoints[index] = new
            return True
        return False
        
    def extend(self, points : list[Point]) -> None:
        while self.extend_endpoint(0, points): pass
        while self.extend_endpoint(1, points): pass
    
###########################################################

fname = "images/m002.sxm"
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

points = np.array([Point(center) for center in w_centers])

origin = points[0]
closest = origin.find_closest(points[1:])
vec = origin.vector_to(closest)

line = Line([origin, closest], vec, 0.2)
line.extend(points)
print(line.get_points_as_ndarray())

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray")
ut.add_toggleable_circles(fig, np.array([ax]), np.roll(w_centers, 1, axis=1), 'b', '2')
ut.add_toggleable_circles(fig, np.array([ax]), np.roll(line.get_points_as_ndarray(), 1, axis=1), 'g', '1')
ax.quiver(points[0].y, points[0].x, *np.roll(vec, 1), color='r', angles='xy', scale_units='xy', scale=1)

plt.show()