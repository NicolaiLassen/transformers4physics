from data.lorenz_data import create_lorenz_sequence
import numpy as np
import h5py


def asd(
    seed=None,
    set_size=256,
    num_steps=[256],
):
    rng = np.random.default_rng(seed)

    sequences = []
    sequence_lengths = []

    for _ in range(set_size):
        seq = rng.random((rng.choice(num_steps), 3, 64, 64)).astype(np.float)
        sequences.append(seq)
        sequence_lengths.append(np.array(len(seq), dtype=int))
    
    return np.array(sequences).reshape((-1, 128, 3, 64, 64))

dataset = asd(
    seed=42,
    num_steps=[128],
    set_size=16,
)

dataset_test = asd(
    seed=24,
    num_steps=[128],
    set_size=4,
)


hf = h5py.File('./tests/koopman_git_2/magnet_data_train.h5', 'w')
hf.create_dataset('dataset_1', data=dataset)
hf.close()

hf = h5py.File('./tests/koopman_git_2/magnet_data_test.h5', 'w')
hf.create_dataset('dataset_1', data=dataset_test)
hf.close()
