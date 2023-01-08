import csv
import numpy as np
from .DatasetLoader import DatasetLoader
from .SpectralDataset import SpectralDataset


class PlasticScannerDatasetLoader(DatasetLoader):
    filepath = None
    material_map = None

    def __init__(self, file: str, material_map: dict):
        self.filepath = file
        self.material_map = material_map

    def load(self) -> SpectralDataset:
        x = []
        y = []
        sample_ids = []
        wavelengths = None
        whitelisted_labels = self.material_map.keys()
        with open(self.filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if wavelengths is None:
                    if len(row) < 4:
                        raise ValueError('Invalid format')
                    wavelengths = np.array(row[3:]).astype(np.uint16)
                    continue
                if len(row) > 3 and (whitelisted_labels is None or row[0] in whitelisted_labels):
                    x.append(np.array(row[3:]).astype(np.float64))
                    y.append(self.material_map[row[0]])
                    sample_ids.append(row[1])
        return SpectralDataset(wavelengths, np.array(x), np.array(y).astype(np.int8), sample_ids=np.array(sample_ids))

    def __eq__(self, other):
        if isinstance(other, PlasticScannerDatasetLoader):
            return self.filepath == other.filepath
        return NotImplemented

