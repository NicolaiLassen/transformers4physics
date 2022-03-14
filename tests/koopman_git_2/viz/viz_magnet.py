"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import os
from typing import Optional

import matplotlib
import matplotlib as mpl
import numpy as np
import torch

mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from viz.viz_model import Viz

Tensor = torch.Tensor

class MicroMagViz(Viz):
    def __init__(self, plot_dir: str = None) -> None:
        super().__init__(plot_dir=plot_dir)
        self.coords = None

    def setCoords(self, coords_path) -> None:
        b = np.load(coords_path)
        b = b.swapaxes(0,1).reshape(3,36,36).swapaxes(1,2)
        self.coords = b
    
    def plotPrediction(self,
        y_pred: Tensor,
        y_target: Tensor,
        plot_dir: str = None,
        epoch: int = None,
        pid: int = 0
    ) -> None:
        assert self.coords is not None, 'Coordinates must be initialized before training/visualization'
        X = self.coords[0]
        Y = self.coords[1]
        # Convert to numpy array
        y_pred = y_pred.detach().cpu().numpy()
        y_target = y_target.detach().cpu().numpy()

        plt.close('all')
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        # mpl.rcParams['figure.dpi'] = 10

        def update_quiver(i, Q, y):
            U = y[i,0]
            V = y[i,1]
            Q.set_UVC(U,V)
            return Q

        # rc('text', usetex=True)
        # Set up figure

        fig, ax = plt.subplots(1,1)
        U = y_pred[0,0]
        V = y_pred[0,1]
        Qpred = plt.quiver(X, Y, U, V, pivot='mid', color='b')
        anim = animation.FuncAnimation(fig, update_quiver, fargs=(Qpred,y_pred), interval=1, blit=False, repeat=False, frames=y_pred.shape[0])

        if(not epoch is None):
            file_name = 'microMagPredict{:d}_{:d}.gif'.format(pid, epoch)
        else:
            file_name = 'microMagPredict{:d}.gif'.format(pid)

        if plot_dir is None:
            plot_dir = self.plot_dir
            f = '{}/{}'.format(plot_dir,file_name)
        else:
            f = '{}/{}'.format(self.plot_dir,plot_dir)
        writergif = animation.PillowWriter(fps=30) 
        anim.save(f, writer=writergif)

    def plotEmbeddingPrediction(self, y_pred: Tensor, y_target: Tensor, plot_dir: str = None, **kwargs) -> None:
        return super().plotEmbeddingPrediction(y_pred, y_target, plot_dir, **kwargs)
