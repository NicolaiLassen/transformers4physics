import h5py
import numpy as np
import matplotlib.pyplot as plt

f1 = h5py.File('./tests/run_time_v0/time_graph_v0_magtense.h5', 'r')
# plt.xkcd()
plt.plot(f1['resolutions'],f1['avgTimeSeconds'])
plt.title('Magtense magnetostatic simulation time based on cube matrix of magnets with varying side lengths')
plt.xlabel('Side length of cube magnet matrix')
plt.ylabel('Simulation time (seconds)')
plt.xticks(f1['resolutions'])
plt.grid()
plt.show()

f2 = h5py.File('./tests/run_time_v0/time_graph_v0_model.h5', 'r')
# plt.xkcd()
plt.plot(f2['resolutions'],f2['avgTimeSeconds'])
plt.title('Model magnetostatic simulation time based on cube matrix of magnets with varying side lengths')
plt.xlabel('Side length of cube magnet matrix')
plt.ylabel('Simulation time (seconds)')
plt.xticks(f2['resolutions'])
plt.grid()
plt.show()