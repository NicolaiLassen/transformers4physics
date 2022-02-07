# %%
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set_theme()
from prism_grid import create_prism_grid

prefix = 'uniform_x'
res = 224

imgin, m, imgout = create_prism_grid(
    rows=2,
    columns=1,
    res=res,
)
imgin, m, imgout = np.array(imgin), np.array(m), np.array(imgout)
m = m - 1
m = m * (-1)
#print(imgin)

#imgin = convertToImage(imgin)
#imgout = convertToImage(imgout)

#%%
showimgin = imgin[0:3]*imgin[3]
showimgout = imgout[0:3]*imgout[3]
def showHeat(images, titles):
    for img,t in zip(images, titles):
        sns.heatmap(img, cmap="mako", mask=m)
        ax = plt.axes()
        ax.set_title('{}_{}'.format(prefix, t))
        #plt.savefig('{}_{}.png'.format(prefix, t))
        plt.show()

showHeat(showimgin[0:3], ['X-magnetization', 'Y-magnetization', 'Z-magnetization'])
showHeat(showimgout[0:3], ['X-field', 'Y-field', 'Z-field'])
#%%
for i in range(1,15):
    print(i)
    print(224/i)