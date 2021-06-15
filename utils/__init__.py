""" Python package for image processing """
import torch.utils.data
from utils.dataset import Dataset
from utils.params import opt


def create_dataset(train_or_test, max_dataset_size=float("inf")):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from utils import create_dataset
        >>> dataset = create_dataset(train_or_test, max_dataset_size)
    """
    data_loader = DataLoader(train_or_test, max_dataset_size)
    data_set = data_loader.load_data()
    return data_set


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class DataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, train_or_test, max_dataset_size=float("inf")):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.max_dataset_size = max_dataset_size
        self.dataset = Dataset(train_or_test)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt['batch_size'],
            # shuffle=not opt.serial_batches,
            shuffle=True,
            num_workers=int(opt['num_threads']))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * opt['batch_size'] >= self.max_dataset_size:
                break
            yield data
