from time import time_ns
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from swin import SwinTransformer


def sim_grid(res=2, iterations=50, seed=None, window_size=8):
    with torch.no_grad():
        if(seed):
            torch.manual_seed(seed)
        times = []
        tf = SwinTransformer(img_size=res, in_chans=4, depths=[2], num_heads=[1], num_classes=4*res*res*res, window_size=window_size).cuda()
        for a in range(iterations):
            randomData = torch.rand((1,4,res,res,res)).cuda()
            timeStart = time_ns()
            _ = tf(randomData)
            timeEnd = time_ns()
            if(not a==0):
                times.append(timeEnd-timeStart)

        return np.array(times)

resolutions = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,64,96,128])
window_sizes = np.array([2,5,3,7,4,3,5,11,6,13,7,5,8,8,8,8,8,8])

avgTimesSeconds = np.zeros((len(resolutions)))
i = 0
for r,ws in zip(resolutions,window_sizes):
    times = sim_grid(res=r, iterations=500, window_size=ws)
    nsAvg = times.mean()
    sAvg = nsAvg*1e-9
    avgTimesSeconds[i] = sAvg
    print(r)
    i = i + 1

hf = h5py.File('./tests/run_time_v0/time_graph_v0_model.h5', 'w')
hf.create_dataset('resolutions', data=resolutions)
hf.create_dataset('avgTimeSeconds', data=avgTimesSeconds)
hf.close()

plt.plot(resolutions,avgTimesSeconds)
plt.show()