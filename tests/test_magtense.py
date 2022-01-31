import numpy as np

from magtense import magtense
from magtense.utils.eval import get_p2p, get_average_magnetic_flux
from magtense.utils.plot import create_plot

import matplotlib.pyplot as plt
import seaborn as sns
plt.set_cmap("cividis")


def prism_grid():
    # Defining grid
    rows = 10
    cols = 10
    margin = 4

    places = [rows + margin, cols + margin, 5]
    area = [1, 1, 0.5]

    filled_positions = []

    for row in range(rows):
        for col in range(cols):
            filled_positions.append([row + int(margin / 2) , col + int(margin / 2), 0])

    # Optional parameters for setup: n_magnets, filled_positions, mag_angles, eval_points, eval_mode, B_rem
    (tiles, points, grid) = magtense.setup(places, area, filled_positions=filled_positions, eval_points=[10, 10, 5])

    # Standard parameters in settings: max_error=0.00001, max_it=500
    (updated_tiles, H) = magtense.run_simulation(tiles, points, grid=grid, plot=True)

    print("Average magnetic field: " + str(get_average_magnetic_flux(H)))
    print("Peak to peak: " + str(get_p2p(H)))

if __name__ == '__main__':
    prism_grid()