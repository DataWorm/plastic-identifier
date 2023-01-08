import copy
import numpy as np
from .DatasetTransformer import DatasetTransformer
from .SpectralDataset import SpectralDataset


class WavelengthFilter(DatasetTransformer):

    def __init__(self, wavelengths : np.ndarray, max_wavelength_offset = 10):
        self.wavelengths = wavelengths
        self.max_offset = max_wavelength_offset

    def transform(self, dataset : SpectralDataset) -> SpectralDataset:
        if np.array_equal(self.wavelengths, dataset.wavelengths):
            return dataset
        x = np.zeros((dataset.X.shape[0], len(self.wavelengths)))
        for column in range(self.wavelengths.shape[0]):
            x[:, column] = self.getOrCalculateWavelengthValues(self.wavelengths[column], dataset)
        transformed_dataset = copy.deepcopy(dataset)
        transformed_dataset.wavelengths = self.wavelengths
        transformed_dataset.X = x
        return transformed_dataset

    def getOrCalculateWavelengthValues(self, wavelength, dataset) -> np.ndarray:
        # find closest wavelength matches below and above target wavelength (or return if identical match was found)
        lowerIndex = -1
        upperIndex = -1
        lowerWavelength = -1
        upperWavelength = -1
        for column in range(dataset.wavelengths.shape[0]):
            currentWavelength = dataset.wavelengths[column]
            if currentWavelength == wavelength:
                return dataset.X[:, column]
            if currentWavelength < wavelength and lowerWavelength < currentWavelength:
                lowerWavelength = currentWavelength
                lowerIndex = column
            if currentWavelength > wavelength and upperWavelength > currentWavelength:
                upperWavelength = currentWavelength
                upperIndex = column
        # ignore closest matches if offset to target wavelength exceeds limits
        if self.max_offset is not None:
            if lowerIndex >= 0 and wavelength - lowerWavelength > self.max_offset:
                lowerIndex = -1
            if upperIndex >= 0 and upperWavelength - wavelength > self.max_offset:
                upperIndex = -1
        # calculate value between upper and lower wavelength measurements if available
        if lowerIndex >= 0 and upperIndex >= 0:
            relative_offset = (wavelength - lowerWavelength) / (upperWavelength - wavelength)
            return dataset.X[:, lowerIndex] + relative_offset * (dataset.X[:, upperIndex] - dataset.X[:, lowerIndex])
        # use closest value if available
        if lowerIndex >= 0:
            return dataset.X[:, lowerIndex]
        if upperIndex >= 0:
            return dataset.X[:, upperIndex]
        raise ValueError('Dataset does not contain measurements close enough to wavelength '+str(wavelength))
