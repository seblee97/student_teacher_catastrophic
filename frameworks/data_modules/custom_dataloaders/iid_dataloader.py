from torch.utils.data import Dataset

import torch

class IIDDataset(Dataset):
    """IID gaussian samples dataset."""

    def __init__(self, dimension: int, repeat: int=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dimension = dimension
        self.repeat = repeat

        if self.repeat:
            self._dataset = torch.randn(repeat, self.input_dimension).to(self.device)

    def __len__(self):
        pass 

    def __getitem__(self, idx):
        if self.repeat:
            input = self.dataset[idx, :]

        return self.dataset