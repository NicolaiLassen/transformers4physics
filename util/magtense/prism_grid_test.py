#%%
import numpy as np
from prism_grid import create_prism_grid

def _checkAllChannelsEq(x, xs, xe, ys, ye):
    for i in range(4):
        if not np.all(x[i, xs:xe, ys:ye] == x[i, xs, ys]):
            return False
    return True

def testImageIn():
    i,h = create_prism_grid(
        rows=1,
        columns=1,
        res=244,
    )

    assert np.all(i[:,0,:] == 0)
    assert np.all(i[:,5,:] == 0)
    assert np.all(i[:,:,0:2] == 0)
    assert np.all(i[:,:,4:6] == 0)
    assert _checkAllChannelsEq(i, 1, 3, 2, 4)
    assert _checkAllChannelsEq(i, 3, 5, 2, 4)

    i,_ = create_prism_grid(
        rows=2,
        columns=2,
        res=4,
    )
    
    assert _checkAllChannelsEq(i, 0, 2, 0, 2)
    assert _checkAllChannelsEq(i, 2, 4, 0, 2)
    assert _checkAllChannelsEq(i, 0, 2, 2, 4)
    assert _checkAllChannelsEq(i, 2, 4, 2, 4)

    print('done')

testImageIn()

# %%
