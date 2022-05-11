import h5py
import math
from matplotlib import pyplot as plt
import numpy as np

base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
# f = h5py.File(base + '\\field_s_state_test_large.h5')
f = h5py.File(base + '\\field_s_state_test_circ.h5')
# f = h5py.File('./field_s_state_rest.h5')
asd = []
for k in f.keys():
    field = np.array( f[k]['field'])
    angle = math.atan2(field[1], field[0])/math.pi*180
    p = (field[0]**2+field[1]**2)**0.5
    if 180-abs(angle) < 44:
    # if abs(angle) < 40:
        print(angle)
        print(p)
        print(k)
        print()
    asd.append(angle)