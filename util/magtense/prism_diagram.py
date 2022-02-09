# %%
from re import M
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns; sns.set_theme()
from prism_grid import create_prism_grid

res = 224

#%matplotlib qt

imgin, m, imgout = create_prism_grid(
    rows=2,
    columns=2,
    res=res,
    plot=True,
    restrict_z=True,
    seed=3,
    uniform_tesla=1.0,
    # uniform_ea=[1,0,0],
)
imgin, m, imgout = np.array(imgin), np.array(m), np.array(imgout)

m = m - 1
m = m * (-1)
#print(imgin)

#imgin = convertToImage(imgin)
#imgout = convertToImage(imgout)

#%%
#%matplotlib inline
def showNorm(imageOut, mask):
    ax = sns.heatmap(imageOut[3], cmap="mako", mask=mask)
    ax.invert_yaxis()
    plt.show()

showNorm(imgout, m)

ax = sns.heatmap(imgin[0], cmap="mako", mask=m)
ax.invert_yaxis()
plt.show()