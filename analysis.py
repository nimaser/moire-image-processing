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
            
            if new_point is None:
                new_point_area = origin.add_vector(-self.vector)
                new_point = new_point_area.pop_closest_within_range(self.points, self.range)
                if new_point is None:
                    self.unincluded_points.append(origin)
                    continue
                
            new_line = Line([origin, new_point], self.tolerance)
            new_line.extend(self.points)
            self.lines.append(new_line)
    
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

fractional_tolerance = 0.4

points = [Point(center) for center in w_centers]

pls0 = ParallelLineSet(points, fractional_tolerance)
pls1 = ParallelLineSet(pls0  , fractional_tolerance)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray")
ut.add_toggleable_circles(fig, np.array([ax]), np.roll(w_centers, 1, axis=1), 'b', '0')

for line, color, key in zip(pls0.lines, ['r', 'g', 'c'], ['1', '2', '3', '4']):
    ut.add_toggleable_circles(fig, np.array([ax]), np.roll(line.get_points_as_ndarray(), 1, axis=1), color, key)

for line, color, key in zip(pls1.lines, ['y', 'k', 'r', 'g', 'c'], ['5', '6', '7', '8']):
    ut.add_toggleable_circles(fig, np.array([ax]), np.roll(line.get_points_as_ndarray(), 1, axis=1), color, key)
    
plt.show()