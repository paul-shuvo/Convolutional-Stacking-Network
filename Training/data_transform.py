from torch.utils import data


class DataTransform(data.Dataset):
    """
    The Dataset class is extended to work with
    PyTorch ecosystem.
    """
    def __init__(self, samples, labels):
        """
        Initializes the DataTransform class.
        :param samples: samples of the dataset.
        :param labels: labels of the dataset.
        """
        self.labels = labels
        self.samples = samples

    def __len__(self):
        """
        Returns the number of samples.
        :return: number of samples
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Returns the item (sample & label) corresponding
        to the index number.
        :param index: index of the sample in the dataset.
        :return: sample & label corresponding to the index
        """
        x = self.samples[index]
        y = self.labels[index]

        return x, y
