from .DatasetTransformer import DatasetTransformer
from .SpectralDataset import SpectralDataset


class NoOpTransformer(DatasetTransformer):

    def transform(self, dataset : SpectralDataset) -> SpectralDataset:
        return dataset
