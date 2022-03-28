
import torch
Tensor = torch.Tensor

class PhysData():
    def __init__(self, data: Tensor, mu: Tensor, std: Tensor):
        self.data = data
        self.mu = mu
        self.std = std


def read_as_embbeding():
    print("TODO")

def read_h5_dataset(
    file_path: str,
    block_size: int,
    batch_size: int = 32,
    stride: int = 5,
    n_data: int = -1,
) -> PhysData:

    return PhysData(torch.rand(2, 14, 3, 32, 32), torch.zeros(3), torch.ones(3))
    assert os.path.isfile(
        file_path), "Training HDF5 file {} not found".format(file_path)

    seq = []
    with h5py.File(file_path, "r") as f:

        n_seq = 0
        for key in f.keys():
            data_series = torch.Tensor(np.array(f[key]))
            # Truncate in block of block_size
            for i in range(0,  data_series.size(0) - block_size + 1, stride):
                seq.append(data_series[i: i + block_size].unsqueeze(0))

            n_seq = n_seq + 1
            if(n_data > 0 and n_seq >= n_data):  # If we have enough time-series samples break loop
                break

    data = torch.cat(seq, dim=0)
    mu = torch.tensor([torch.mean(data[:, :, 0]), torch.mean(
        data[:, :, 1]), torch.mean(data[:, :, 2])])
    std = torch.tensor([torch.std(data[:, :, 0]), torch.std(
        data[:, :, 1]), torch.std(data[:, :, 2])])

    if data.size(0) < batch_size:
        batch_size = data.size(0)

    return PhysData(data, mu, std)
