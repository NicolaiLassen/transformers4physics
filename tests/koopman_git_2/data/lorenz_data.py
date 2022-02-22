import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt

class LorenzData(torch.utils.data.Dataset):
    def __init__(self, sequences_in=None, sequence_lengths=None):
        self.x = torch.tensor(sequences_in, dtype=torch.float)
        self.l = sequence_lengths

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # return self.x[idx], self.l[idx]
        return {"states": self.x[idx]}


def get_lorenz_statistics(sequences):
    sequences = np.array(sequences)
    print(sequences.shape)
    x, y, z = [], [], []
    for seq in sequences:
        x.extend(seq[:, 0])
        y.extend(seq[:, 1])
        z.extend(seq[:, 2])
    x, y, z = np.array(x), np.array(y), np.array(z)
    return x.mean(), y.mean(), z.mean(), x.std(), y.std(), z.std()


def normalize_lorenz_seq(seq, myx, myy, myz, stdx, stdy, stdz):
    seq = np.array(seq)
    oldShape = seq.shape
    seq = seq.reshape(-1,3)
    x, y, z = seq[:, 0], seq[:, 1], seq[:, 2]
    x = (x-myx)/stdx
    y = (y-myy)/stdy
    z = (z-myz)/stdz
    seq[:, 0], seq[:, 1], seq[:, 2] = x, y, z
    seq = seq.reshape(oldShape)
    return seq


def denormalize_lorenz_seq(seq, myx, myy, myz, stdx, stdy, stdz):
    seq = np.array(seq)
    oldShape = seq.shape
    seq = seq.reshape(-1,3)
    x, y, z = seq[:, 0], seq[:, 1], seq[:, 2]
    x = stdx*x+myx
    y = stdy*y+myy
    z = stdz*z+myz
    seq[:, 0], seq[:, 1], seq[:, 2] = x, y, z
    seq = seq.reshape(oldShape)
    return seq


def normalize_lorenz_sequences(sequences):
    myx, myy, myz, stdx, stdy, stdz = get_lorenz_statistics(sequences)
    n_seq = []
    for seq in sequences:
        n_seq.append(normalize_lorenz_seq(
            seq, myx, myy, myz, stdx, stdy, stdz))
    return n_seq, myx, myy, myz, stdx, stdy, stdz


def plot_lorenz(seq, title="Lorenz Attractor"):
    x = [s[0] for s in seq]
    y = [s[1] for s in seq]
    z = [s[2] for s in seq]

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

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


def create_lorenz_sequence(x, y, z, steps=10000, dt=0.01, include_vel=False):
    seq = np.empty((steps, 6 if include_vel else 3))
    if(include_vel):
        x_dot, y_dot, z_dot = _lorenz(x, y, z)
        seq[0] = [x, y, z, x_dot, y_dot, z_dot]
        for i in range(1, steps):
            x = seq[i-1, 0] + (seq[i-1, 3]*dt)
            y = seq[i-1, 1] + (seq[i-1, 4]*dt)
            z = seq[i-1, 2] + (seq[i-1, 5]*dt)
            x_dot, y_dot, z_dot = _lorenz(x, y, z)
            seq[i] = [
                x,
                y,
                z,
                x_dot,
                y_dot,
                z_dot,
            ]
    else:
        seq[0] = [x, y, z]
        for i in range(1, steps):
            x_dot, y_dot, z_dot = _lorenz(
                seq[i-1, 0], seq[i-1, 1], seq[i-1, 2])
            seq[i] = [
                seq[i-1, 0] + (x_dot*dt),
                seq[i-1, 1] + (y_dot*dt),
                seq[i-1, 2] + (z_dot*dt),
            ]
    return seq


def average_lorenz_sequences(sequences):
    return np.average(sequences, axis=0)


def create_lorenz_dataset(
    seed=None,
    set_size=200,
    x=[-20, 20],
    y=[-20, 20],
    z=[10, 40],
    dt=0.01,
    num_steps=[2048, 64, 256],
    normalize=True,
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

    if(normalize):
        sequences, myx, myy, myz, stdx, stdy, stdz = normalize_lorenz_sequences(
            sequences)
        return LorenzData(sequences_in=sequences, sequence_lengths=sequence_lengths), myx, myy, myz, stdx, stdy, stdz

    return LorenzData(sequences_in=sequences, sequence_lengths=sequence_lengths)


def create_lorenz_data_loader(
    dataset,
    batch_size=1,
    use_cuda=True,
):
    # def c(data):
    #     _, l = zip(*data)
    #     max_len = max(l)
    #     n_features = data[0][0].size(1)
    #     sequences = torch.zeros((len(data), max_len, n_features))
    #     lengths = torch.tensor(l)

    #     for i in range(len(data)):
    #         j, k = data[i][0].size(0), data[i][0].size(1)
    #         sequences[i] = torch.cat(
    #             [data[i][0], torch.zeros((max_len - j, k))])
    #     if(use_cuda):
    #         return sequences.float().cuda(), lengths.long().cuda()
    #     else:
    #         return sequences.float(), lengths.long()

    def c(data):
        x_data_tensor = torch.stack([d['states'] for d in data])
        return {'states': x_data_tensor}

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # collate_fn=c,
        collate_fn = c
    )


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    asd = create_lorenz_sequence(10, 10, 10, steps=2)
    asd2 = create_lorenz_sequence(0, 0, 10, steps=2)
    asd3 = create_lorenz_sequence(0, 0, 10, steps=2)
    asd4 = create_lorenz_sequence(0, 0, 10, steps=2)
    print(asd)
    plot_lorenz(asd)
    x, y, z, a, b, c = get_lorenz_statistics([asd, asd2, asd3, asd4])
    asd = normalize_lorenz_seq(asd, x, y, z, a, b, c)
    print(asd)
    plot_lorenz(asd)
    asd = denormalize_lorenz_seq(asd, x, y, z, a, b, c)
    print(asd)
    plot_lorenz(asd)
