import copy
import numpy as np
from .DatasetTransformer import DatasetTransformer
from .SpectralDataset import SpectralDataset


class SpectralonCalibrationTransformer(DatasetTransformer):

    SPECTRALON_ID = 0

    def transform(self, dataset : SpectralDataset) -> SpectralDataset:
        filter_mask = dataset.y == SpectralonCalibrationTransformer.SPECTRALON_ID
        if not filter_mask.any():
            raise ValueError("No spectralon entries found in dataset for calibration")
        averaged_measurements = np.mean(dataset.X[filter_mask], axis=0)
        transformed_dataset = copy.deepcopy(dataset)
        transformed_dataset.X = dataset.X/averaged_measurements
        return transformed_dataset
