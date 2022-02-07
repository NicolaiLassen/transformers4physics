# %%
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set_theme()

#from util.magtense.prism_grid import create_prism_grid

def convertToImage(img):
    w, h = img[0].shape
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i,channel in enumerate(img[0:3,:,:]):
        for j in range(w):
            for k in range(h):
                data[k, j, i] = channel[j,k]*255
    return data

res = 128
imgin, m, imgout = create_prism_grid(
    rows=3,
    columns=3,
    res=res,
)
imgin, m, imgout = np.array(imgin), np.array(m), np.array(imgout)
m = m - 1
m = m * (-1)
#print(imgin)

#imgin = convertToImage(imgin)
#imgout = convertToImage(imgout)

#%%
print(imgin)
#%%
showimgin = imgin[0:3]*imgin[3]
showimgout = imgout[0:3]*imgout[3]
def showHeat(images):
    for img in images:
        sns.heatmap(img, cmap="mako", mask=m)
        plt.show()

showHeat(showimgin[0:3])
showHeat(showimgout[0:3])
#%%
for i in range(1,15):
    print(i)
    print(224/i)