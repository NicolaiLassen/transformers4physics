import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt


class TrajectoryDataSet(torch.utils.data.Dataset):
    def __init__(self, sequences_in=None, sequence_lengths=None):
        self.x = sequences_in
        self.l = sequence_lengths

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.l[idx]


def draw_trajectory(seq, only_pos = False):
    x = []
    y = []
    if only_pos:
        x = [s[0] for s in seq]
        y = [s[1] for s in seq]
    else:
        x = [s[2] for s in seq]
        y = [s[3] for s in seq]
    plt.plot(x, y)
    plt.show()


def create_trajectory_seq(init_vel_x, init_vel_y, init_x, init_y, g, timescale=0.05):
    seq = []
    seq.append([init_vel_x, init_vel_y, init_x, init_y])
    while(seq[-1][3] >= 0):
        seq.append([
            seq[-1][0],
            seq[-1][1]-g*timescale,
            seq[-1][2]+seq[-1][0]*timescale,
            seq[-1][3]+seq[-1][1]*timescale,
        ])
    return seq


def create_trajectory_dataset(
    seed=None,
    set_size=512,
    init_vel_x=[5, 25],
    init_vel_y=[5, 25],
    init_x=[0, 10],
    init_y=[0, 10],
    g=9.82,
    timescale=0.05,
):
    rng = np.random.default_rng(seed)
    sequences = []
    sequence_lengths = []
    for _ in range(set_size):
        seq = create_trajectory_seq(
            rng.random()*(init_vel_x[1]-init_vel_x[0])+init_vel_x[0],
            rng.random()*(init_vel_y[1]-init_vel_y[0])+init_vel_y[0],
            rng.random()*(init_x[1]-init_x[0])+init_x[0],
            rng.random()*(init_y[1]-init_y[0])+init_y[0],
            g,
            timescale=timescale,
        )
        sequences.append(torch.tensor(seq, dtype=torch.float))
        sequence_lengths.append(torch.tensor(len(seq), dtype=torch.long))
    return TrajectoryDataSet(sequences_in=sequences, sequence_lengths=sequence_lengths)


def create_trajectory_data_loader(
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
