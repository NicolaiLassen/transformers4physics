import numpy as np
import h5py

def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

if __name__ == '__main__':
    num_seq = 55

    rng = np.random.default_rng(500)
    r = rng.random((num_seq,3,36,36,1)) * 2 - 1
    r2 = rng.random((num_seq,3,32,32,1)) * 2 - 1
    fields = np.zeros((num_seq,3))
    for i in range(num_seq):
        fields[i,0:2] = (num_seq) * rng.random((2)) - 25
    r_18x18 = np.zeros((num_seq,3,18,18,1))
    r_9x9 = np.zeros((num_seq,3,9,9,1))
    for i in range(num_seq):
        r_18x18[i,0,:,:,0] = pooling(r[i,0,:,:,0],(2,2),method='mean')
        r_18x18[i,1,:,:,0] = pooling(r[i,1,:,:,0],(2,2),method='mean')
        r_18x18[i,2,:,:,0] = pooling(r[i,2,:,:,0],(2,2),method='mean')

        r_9x9[i,0,:,:,0] = pooling(r[i,0,:,:,0],(4,4),method='mean')
        r_9x9[i,1,:,:,0] = pooling(r[i,1,:,:,0],(4,4),method='mean')
        r_9x9[i,2,:,:,0] = pooling(r[i,2,:,:,0],(4,4),method='mean')

        
    r_16x16 = np.zeros((num_seq,3,16,16,1))
    r_8x8 = np.zeros((num_seq,3,8,8,1))
    for i in range(num_seq):
        r_16x16[i,0,:,:,0] = pooling(r2[i,0,:,:,0],(2,2),method='mean')
        r_16x16[i,1,:,:,0] = pooling(r2[i,1,:,:,0],(2,2),method='mean')
        r_16x16[i,2,:,:,0] = pooling(r2[i,2,:,:,0],(2,2),method='mean')

        r_8x8[i,0,:,:,0] = pooling(r2[i,0,:,:,0],(4,4),method='mean')
        r_8x8[i,1,:,:,0] = pooling(r2[i,1,:,:,0],(4,4),method='mean')
        r_8x8[i,2,:,:,0] = pooling(r2[i,2,:,:,0],(4,4),method='mean')
        
    hf = h5py.File('./starting_states_36x36_train.h5', 'w')
    hf.create_dataset('states', data=r[:50])
    hf.create_dataset('fields', data=fields[:50])
    hf.close()    
    hf = h5py.File('./starting_states_18x18_train.h5', 'w')
    hf.create_dataset('states', data=r_18x18[:50])
    hf.create_dataset('fields', data=fields[:50])
    hf.close()    
    hf = h5py.File('./starting_states_9x9_train.h5', 'w')
    hf.create_dataset('states', data=r_9x9[:50])
    hf.create_dataset('fields', data=fields[:50])
    hf.close()
    hf = h5py.File('./starting_states_32x32_train.h5', 'w')
    hf.create_dataset('states', data=r2[:50])
    hf.create_dataset('fields', data=fields[:50])
    hf.close()
    hf = h5py.File('./starting_states_16x16_train.h5', 'w')
    hf.create_dataset('states', data=r_16x16[:50])
    hf.create_dataset('fields', data=fields[:50])
    hf.close()
    hf = h5py.File('./starting_states_8x8_train.h5', 'w')
    hf.create_dataset('states', data=r_8x8[:50])
    hf.create_dataset('fields', data=fields[:50])
    hf.close()

    hf = h5py.File('./starting_states_36x36_test.h5', 'w')
    hf.create_dataset('states', data=r[50:])
    hf.create_dataset('fields', data=fields[50:])
    hf.close()    
    hf = h5py.File('./starting_states_18x18_test.h5', 'w')
    hf.create_dataset('states', data=r_18x18[50:])
    hf.create_dataset('fields', data=fields[50:])
    hf.close()    
    hf = h5py.File('./starting_states_9x9_test.h5', 'w')
    hf.create_dataset('states', data=r_9x9[50:])
    hf.create_dataset('fields', data=fields[50:])
    hf.close()
    hf = h5py.File('./starting_states_32x32_test.h5', 'w')
    hf.create_dataset('states', data=r2[50:])
    hf.create_dataset('fields', data=fields[50:])
    hf.close()
    hf = h5py.File('./starting_states_16x16_test.h5', 'w')
    hf.create_dataset('states', data=r_16x16[50:])
    hf.create_dataset('fields', data=fields[50:])
    hf.close()
    hf = h5py.File('./starting_states_8x8_test.h5', 'w')
    hf.create_dataset('states', data=r_8x8[50:])
    hf.create_dataset('fields', data=fields[50:])
    hf.close()
