from data.lorenz_data import create_lorenz_sequence
import numpy as np
import h5py

def asd(
    seed=None,
    set_size=200,
    x=[-20, 20],
    y=[-20, 20],
    z=[10, 40],
    dt=0.01,
    num_steps=[256],
    include_vel=False,
):
    rng = np.random.default_rng(seed)
    sequences = []
    sequence_lengths = []
    for _ in range(set_size):
        seq = create_lorenz_sequence(
            rng.random()*(x[1]-x[0])+x[0],
            rng.random()*(y[1]-y[0])+y[0],
            rng.random()*(z[1]-z[0])+z[0],
            steps=rng.choice(num_steps),
            dt=dt,
        )
        sequences.append(np.array(seq, dtype=np.float64).reshape(
            6 if include_vel else 3, -1))
        sequence_lengths.append(np.array(len(seq), dtype=int))

    return np.array(sequences).reshape((-1,3))

dataset = asd(
    seed=42,
    dt=0.01,
    num_steps=[512],
    set_size=128,
)
dataset_test = asd(
    seed=24,
    dt=0.01,
    num_steps=[512],
    set_size=32,
)
hf = h5py.File('./tests/koopman_git_2/lorenz_data_train.h5', 'w')
hf.create_dataset('dataset_1', data=dataset)
hf.close()

hf = h5py.File('./tests/koopman_git_2/lorenz_data_test.h5', 'w')
hf.create_dataset('dataset_1', data=dataset_test)
hf.close()