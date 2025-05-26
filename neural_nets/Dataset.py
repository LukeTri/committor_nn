import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Select sample
        X = self.samples[index]
        y = self.labels[index]

        return X, y
