import pickle
import gzip
import numpy as np

class Dataloader:
    def __init__(self, path: str):
        self.path = path

    def _load_data(self) -> any:
        with gzip.open(self.path, 'rb') as file:
            return pickle.load(file, encoding='latin1')

    def _vectorize(self, j: int):
        arr = np.zeros((10,1))
        arr[j] = 1.0
        return arr

    def load(self):
        train_data, validation_data, test_data = self._load_data()

        training_inputs, training_results  = [np.reshape(x, (784, 1)) for x in train_data[0]], [self._vectorize(y) for y in train_data[1]]
        training_data = list(zip(training_inputs, training_results))
        
        validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
        validation_data = list(zip(validation_inputs, validation_data[1]))
        
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = list(zip(test_inputs, test_data[1]))
        
        return (training_data, validation_data, test_data)

