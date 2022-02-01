import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from magtense.utils.plot import create_plot


def sample_check(field, updated_tiles=None, points=None, v_max=1, filename=f'foo_{os.getpid()}', cube=False, structure=False):
    plotpath = os.path.dirname(os.path.abspath(__file__)) + '/../plots/sample_check'
    if not os.path.exists(plotpath): os.makedirs(plotpath)
    plt.clf()
    labels = ['Hx-field', 'Hy-field', 'Hz-field']
    nrows = 3 if cube else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=3, sharex=True, sharey=True, figsize=(15,10))

    if cube:
        for i, z in enumerate([0, 1, 2]):
            for j, comp in enumerate(field[:,:,:,z]):
                ax = axes.flat[i * 3 + j]
                im = ax.imshow(comp, cmap='bwr', norm=colors.Normalize(vmin=-v_max, vmax=v_max), origin="lower")
                ax.set_title(labels[j] + f'@{z+1}')

    else:
        for i, comp in enumerate(field):
            ax = axes.flat[i]
            im = ax.imshow(comp, cmap='bwr', norm=colors.Normalize(vmin=-v_max, vmax=v_max), origin="lower")
            ax.set_title(labels[i])
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.345, 0.015, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig(f'{plotpath}/{filename}.png', bbox_inches='tight')

    # Plot magnetic structure
    if structure: create_plot(updated_tiles, points, field, v_max=2*v_max, filename=filename)