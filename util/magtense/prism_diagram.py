# %%
from re import M
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns; sns.set_theme()
from prism_grid import create_prism_grid

res = 244

# %matplotlib qt

imgin, m, imgout = create_prism_grid(
    rows=2,
    columns=2,
    res=res,
    plot=True,
    restrict_z=True,
    seed=48,
    uniform_tesla=1.0,
    uniform_ea=[0,0,-1],
)
imgin, m, imgout = np.array(imgin), np.array(m), np.array(imgout)

m = m - 1
m = m * (-1)
#print(imgin)

#imgin = convertToImage(imgin)
#imgout = convertToImage(imgout)

#%%
# %matplotlib inline
def showNorm(imageOut, mask):
    sns.heatmap(imageOut[3], cmap="mako", mask=mask)
    plt.show()

showNorm(imgout, m)