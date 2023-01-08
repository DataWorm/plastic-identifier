import numpy as np


class SpectralDataset:
    wavelengths = []
    sample_ids = None
    X = []
    y = []

    def __init__(self, wavelengths : np.ndarray, X : np.ndarray, y : np.ndarray, sample_ids=None):
        self.wavelengths = wavelengths
        self.X = X
        self.y = y
        self.sample_ids = sample_ids
        assert wavelengths.ndim == 1
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0], "Different amount of samples in X and y"
        assert X.shape[1] == wavelengths.shape[0], "Number of values in X do not match number of wavelengths"
        assert sample_ids is None or X.shape[0] == len(
            sample_ids), "Number of sample ids does not match number of samples"
