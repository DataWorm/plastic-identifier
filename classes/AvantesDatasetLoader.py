import csv
import numpy as np
from .DatasetLoader import DatasetLoader
from .SpectralDataset import SpectralDataset


class AvantesDatasetLoader(DatasetLoader):
    FIELD_SAMPLE = 1
    FIELD_DARK = 2
    FIELD_REFERENCE = 3
    FIELD_REFLECTANCE = 4

    def __init__(self, directory: str, material_map: dict, measurement_field: int = FIELD_REFLECTANCE):
        self.directory = directory if directory.endswith('/') else directory + '/'
        self.material_map = material_map
        self.measurement_field = measurement_field

    def load(self) -> SpectralDataset:
        x = []
        y = []
        sample_ids = []
        wavelengths = None
        whitelisted_labels = self.material_map.keys()
        file_type_map = self.load_file_type_list()
        for file_id, material_type in file_type_map.items():
            if whitelisted_labels is not None and material_type not in whitelisted_labels:
                continue
            wavelengths_list = []
            values = []
            with open(self.directory + file_id + '.txt', 'r') as file:
                reader = csv.reader(file, delimiter=';')
                skip_header = True
                for row in reader:
                    if skip_header:
                        if len(row) == 0 or (len(row) == 1 and row[0].strip() == ''):
                            skip_header = False
                        continue
                    if len(row) == 5:
                        wavelengths_list.append(row[0].replace(',', '.'))
                        values.append(row[self.measurement_field].lstrip().replace(',', '.'))
            y.append(self.material_map[material_type])
            sample_ids.append(file_id)
            x.append(np.array(values).astype(np.float64))
            if wavelengths is None:
                wavelengths = wavelengths_list
            if wavelengths != wavelengths_list:
                raise ValueError("Expect all measurements in dataset to use the same wavelengths")
        return SpectralDataset(np.array(wavelengths).astype(np.float32), np.array(x), np.array(y).astype(np.int8),
                               sample_ids=np.array(sample_ids))

    def load_file_type_list(self):
        skip_headline = True
        file_type_map = {}
        whitelisted_labels = self.material_map.keys()
        with open(self.directory + 'types.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if skip_headline:
                    skip_headline = False
                    continue
                if len(row) == 2 and (whitelisted_labels is None or row[1] in whitelisted_labels):
                    file_type_map[row[0]] = row[1]
        return file_type_map

    def __eq__(self, other):
        if isinstance(other, AvantesDatasetLoader):
            return self.directory == other.directory
        return NotImplemented
