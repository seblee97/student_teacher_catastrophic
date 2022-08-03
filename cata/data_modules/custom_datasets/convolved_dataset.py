import torch
from torch.utils.data import Dataset


class ConvolvedDataset(Dataset):
    """IID gaussian samples dataset."""

    def __init__(self, dimension: int, num_datapoints: int, repeat: bool = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.dimension = dimension
        self.repeat = repeat

        self._dataset = torch.randn(num_datapoints, self.dimension)

        self.dataset_looped = False

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if idx == 0 and not self.dataset_looped:
            self.dataset_looped = True
        elif idx == 0 and not self.repeat and self.dataset_looped:
            raise StopIteration(
                "This IID dataset has already been looped through but repeat is off. Please check")
        return self._dataset[idx]
