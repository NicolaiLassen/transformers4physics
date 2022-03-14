import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

a = np.load('magtense_micro_test\cube36_3d.npy').squeeze()
a = np.swapaxes(a, 1, 2)
a = a.reshape(500, 3, 36, 36)
a = np.swapaxes(a,2,3)


b = np.load('magtense_micro_test\cube36_3d_coord.npy')
b = np.swapaxes(b, 0, 1)
b = b.reshape(3, 36, 36)
b = np.swapaxes(b,1,2)

X = b[0]
Y = b[1]
U = a[0,0]
V = a[0,1]

fig, ax = plt.subplots(1,1)
Q = plt.quiver(X, Y, U, V, pivot='mid', color='b')

def update_quiver(num, Q, _):
    U = a[num,0]
    V = a[num,1]
    Q.set_UVC(U,V)
    return Q

anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,1), interval=1, blit=False, repeat=False, frames=500)

fig.tight_layout()
plt.show()