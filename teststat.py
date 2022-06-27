from matplotlib import pyplot as plt
import numpy as np

# let us generate fake test data
x = np.arange(4)
y1 = np.array([0,1,1,0])
y2 = np.array([1,0,0,1])

def abc(y1,y2):
    z = y1-y2
    dx = x[1:] - x[:-1]
    cross_test = np.sign(z[:-1] * z[1:])

    x_intersect = x[:-1] - dx / (z[1:] - z[:-1]) * z[:-1]
    dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]

    areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx # signs of both z are same
    areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])

    areas = np.where(cross_test < 0, areas_neg, areas_pos)
    total_area = np.sum(areas)
    
    return total_area