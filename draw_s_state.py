from matplotlib.pylab import figure
import matplotlib.pyplot as plt
import h5py
import numpy as np

f = h5py.File('./s_state.h5')
s_state = np.array(f['s_state'])


width = 0.002
headwidth = 2
headlength = 5
figure(figsize=(16,8),dpi=140)
plt.quiver(s_state[0].T, s_state[1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
plt.axis("scaled")
# plt.ylabel('y', fontsize=32, rotation = 0)
# plt.xlabel('x', fontsize=32)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\s_state.png', format='png', bbox_inches='tight')