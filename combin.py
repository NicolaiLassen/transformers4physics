import h5py
import numpy as np

f1 = h5py.File('./prob1.h5')
prob1 = f1['0']
f2 = h5py.File('./prob2.h5')
prob2 = f2['0']
seq1 = np.array(prob1['sequence'])
field1 = np.array(prob1['field'])
seq2 = np.array(prob2['sequence'])
field2 = np.array(prob2['field'])

f1.close()
f2.close()
fn = h5py.File('./problem4.h5','w')
g0 = fn.create_group('0')
g1 = fn.create_group('1')
g0.create_dataset('sequence', data=seq1)
g0.create_dataset('field', data=field1)
g1.create_dataset('sequence', data=seq2)
g1.create_dataset('field', data=field2)
fn.close()