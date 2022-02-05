#%%
import numpy as np
from prism_grid import create_prism_grid

def testImageIn():
    i,_ = create_prism_grid(
        rows=1,
        columns=2,
        res=6,
    )
    
    print(i.shape)
    assert np.all(i[0,:,:] == 0)
    assert np.all(i[:,0:2,:] == 0)
    assert np.all(i[:,4:6,:] == 0)
    assert np.all(i[1:3,2:4,:] == i[1,2,:])
    assert np.all(i[3:5,2:4,:] == i[3,2,:])

    i,_ = create_prism_grid(
        rows=2,
        columns=2,
        res=4,
    )
    
    print(i.shape)
    assert np.all(i[0:2,0:2,:] == i[0,0,:])
    assert np.all(i[0:2,2:4,:] == i[0,2,:])
    assert np.all(i[2:4,0:2,:] == i[2,0,:])
    assert np.all(i[2:4,2:4,:] == i[2,2,:])

    print('done')

testImageIn()
