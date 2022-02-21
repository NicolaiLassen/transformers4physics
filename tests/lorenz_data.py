import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt


class LorenzData(torch.utils.data.Dataset):
    def __init__(self, sequences_in=None, sequence_lengths=None):
        self.x = sequences_in
        self.l = sequence_lengths

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.l[idx]


def plot_lorenz(seq):
    x = [s[0] for s in seq]
    y = [s[1] for s in seq]
    z = [s[2] for s in seq]

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()


def _lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def create_lorenz_sequence(x, y, z, steps=10000, dt=0.01):
    seq = np.empty((steps+1, 3))
    seq[0] = [x, y, z]
    for i in range(steps):
        x_dot, y_dot, z_dot = _lorenz(seq[i, 0], seq[i, 1], seq[i, 2])
        seq[i+1] = [
            seq[i, 0] + (x_dot*dt),
            seq[i, 1] + (y_dot*dt),
            seq[i, 2] + (z_dot*dt),
        ]
    return seq


def create_lorenz_dataset(
    seed=None,
    set_size=200,
    x=[-20, 20],
    y=[-20, 20],
    z=[10, 40],
    dt=0.01,
    num_steps=[2047, 63, 255],
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
        sequences.append(torch.tensor(seq, dtype=torch.float))
        sequence_lengths.append(torch.tensor(len(seq), dtype=torch.long))
    
    return LorenzData(sequences_in=sequences, sequence_lengths=sequence_lengths)

def create_lorenz_data_loader(
    dataset,
    batch_size=1,
    use_cuda=True,
):
    def c(data):
        _, l = zip(*data)
        max_len = max(l)
        n_features = data[0][0].size(1)
        sequences = torch.zeros((len(data), max_len, n_features))
        lengths = torch.tensor(l)

        for i in range(len(data)):
            j, k = data[i][0].size(0), data[i][0].size(1)
            sequences[i] = torch.cat(
                [data[i][0], torch.zeros((max_len - j, k))])
        if(use_cuda):
            return sequences.float().cuda(), lengths.long().cuda()
        else:
            return sequences.float(), lengths.long()

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=c,
    )
