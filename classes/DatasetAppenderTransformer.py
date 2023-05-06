import copy
import numpy as np
from .DatasetTransformer import DatasetTransformer
from .SpectralDataset import SpectralDataset


class DatasetAppenderTransformer(DatasetTransformer):

    def __init__(self, dataset: SpectralDataset = None):
        self.dataset = dataset

    def transform(self, ds: SpectralDataset) -> SpectralDataset:
        if self.dataset is None:
            self.dataset = ds
            return self.dataset
        if not np.array_equal(self.dataset.wavelengths, ds.wavelengths):
            raise ValueError("No spectralon entries found in dataset for calibration")
        X = np.concatenate((self.dataset.X, ds.X))
        y = np.concatenate((self.dataset.y, ds.y))
        sample_ids = []
        sample_ids.extend(self.dataset.sample_ids)
        sample_ids.extend(ds.sample_ids)
        self.dataset = SpectralDataset(self.dataset.wavelengths, X, y, sample_ids=np.array(sample_ids))
        return self.dataset
