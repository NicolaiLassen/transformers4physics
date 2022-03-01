from time import time_ns
from magtense import magtense
import numpy as np
import matplotlib.pyplot as plt
import h5py


def sim_grid(res=2, iterations=50, seed=None):
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(iterations):
        points = np.zeros((res, res, res, 3))
        for i in range(res):
            for j in range(res):
                for k in range(res):
                    points[i, j, k, :] = np.array([0.5+j, 0.5, 0.5])
        points = points.reshape((res*res*res, 3))
        tiles = magtense.Tiles(res**3)
        tiles.set_tile_type(2)
        tiles.set_size([1, 1, 1])
        for d in range(res):
            for c in range(res):
                for r in range(res):
                    i = r+c*res+d*res*res
                    offset = [0.5+c, 0.5+r, 0.5+d]
                    tiles.set_offset_i(offset, i)
                    tiles.set_center_pos_i(offset, i)
                    ea = [
                        rng.random()*2-1,
                        rng.random()*2-1,
                        rng.random()*2-1,
                    ]
                    tiles.set_easy_axis_i(ea, i)
                    tiles.set_remanence_i((rng.random()*0.5+1), i)

        timeStart = time_ns()
        _ = magtense.run_simulation(tiles, points, console=False)
        timeEnd = time_ns()
        times.append(timeEnd-timeStart)

    return np.array(times)

resolutions = np.arange(1,16,1)
print(resolutions)
avgTimesSeconds = np.zeros((len(resolutions)))
i = 0
for r in resolutions:
    times = sim_grid(res=r, iterations=5 if r < 10 else 3)
    nsAvg = times.mean()
    sAvg = nsAvg*1e-9
    avgTimesSeconds[i] = sAvg
    print(r)
    i = i + 1

hf = h5py.File('./tests/run_time_v0/time_graph_v0_magtense.h5', 'w')
hf.create_dataset('resolutions', data=resolutions)
hf.create_dataset('avgTimeSeconds', data=avgTimesSeconds)
hf.close()

plt.plot(resolutions,avgTimesSeconds)
plt.show()