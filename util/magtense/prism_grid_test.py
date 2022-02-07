#%%
import numpy as np
#from prism_grid import create_prism_grid

def _checkAllChannelsEq(x, xs, xe, ys, ye):
    for i in range(4):
        if not np.all(x[i, xs:xe, ys:ye] == x[i, xs, ys]):
            return False
    return True

def testImageIn():
    i,m,_ = create_prism_grid(
        rows=1,
        columns=2,
        res=6,
    )
    i = np.array(i)
    print(i)

    assert np.all(i[:,0,:] == 0)
    assert np.all(i[:,5,:] == 0)
    assert np.all(i[:,0:2,:] == 0)
    assert np.all(i[:,0:2,:] == 0)
    assert _checkAllChannelsEq(i, 2, 4, 1, 3)
    assert _checkAllChannelsEq(i, 2, 4, 3, 5)

    i,m,_ = create_prism_grid(
        rows=2,
        columns=2,
        res=4,
    )
    i = np.array(i)
    
    assert _checkAllChannelsEq(i, 0, 2, 0, 2)
    assert _checkAllChannelsEq(i, 2, 4, 0, 2)
    assert _checkAllChannelsEq(i, 0, 2, 2, 4)
    assert _checkAllChannelsEq(i, 2, 4, 2, 4)

    print('done')

testImageIn()

# %%
