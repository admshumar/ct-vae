import numpy as np
import os
from time import time


class GenericLoader:
    """
   A quick and dirty data loader. Assumes that data are NumPy arrays and are named x_train, x_val, x_test, y_train,
   y_val, and y_test.
   """

    def __init__(self, data_directory):
        if __name__ == "__main__":
            directory = os.path.abspath(os.path.join(os.getcwd(), '../data', data_directory))
        else:
            directory = os.path.abspath(os.path.join(os.getcwd(), 'data', data_directory))
        self.directory = directory
        self.x_train = np.load(os.path.join(directory, 'x_train.npy'))
        self.x_val = np.load(os.path.join(directory, 'x_val.npy'))
        self.x_test = np.load(os.path.join(directory, 'x_test.npy'))
        self.y_train = np.load(os.path.join(directory, 'y_train.npy'))
        self.y_val = np.load(os.path.join(directory, 'y_val.npy'))
        self.y_test = np.load(os.path.join(directory, 'y_test.npy'))

    def load(self):
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test


class MNISTLoader:
    def __init__(self, data_directory, angle_threshold=180):
        if __name__ == "__main__":
            directory = os.path.abspath(os.path.join(os.getcwd(), '../data/mnist', data_directory))
        else:
            directory = os.path.abspath(os.path.join(os.getcwd(), 'data/mnist', data_directory))
        self.directory = directory
        self.angle_threshold = angle_threshold

    def get_error_string(self, digit, number_of_rotations, angle_of_rotation):
        error_string = f'Dataset with parameters {digit}, {number_of_rotations}, {angle_of_rotation}'
        if self.angle_threshold != 180:
            error_string += f' (Threshold: {self.angle_threshold})'
        error_string += ' not found!'
        return error_string

    def get_file(self, digit, number_of_rotations, angle_of_rotation, label=False, angles=False):
        if label:
            file = f'labels=[{digit}]_n_rot={number_of_rotations}_angle={angle_of_rotation}'
        elif angles:
            file = f'angles=[{digit}]_n_rot={number_of_rotations}_angle={angle_of_rotation}'
        else:
            file = f'digits=[{digit}]_n_rot={number_of_rotations}_angle={angle_of_rotation}'
        if self.angle_threshold != 180:
            file += f'_threshold={self.angle_threshold}'
        file += '.npy'

        file_path = os.path.join(self.directory, file)
        print(file_path)
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            raise FileNotFoundError(self.get_error_string(digit, number_of_rotations, angle_of_rotation))

    def load(self, list_of_digits, number_of_rotations, angle_of_rotation, label=False, angles=False):
        t0 = time()
        print("Loading data.")
        data_tuple = tuple(self.get_file(digit, number_of_rotations, angle_of_rotation, label=label, angles=angles)
                           for digit in list_of_digits)
        data = np.concatenate(data_tuple)
        t1 = time()
        t = t1 - t0
        print(f"Data loaded in {t} seconds.")
        return data


class ExperimentFileLoader:
    def __init__(self, model_name, experiment):
        self.model_name = model_name
        self.hyper_parameter_string = experiment
        self.directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'experiments',
                                                      model_name,
                                                      experiment))

    def load_true_labels(self):
        file = os.path.join(self.directory, 'y_true.npy')
        return np.load(file)

    def load_predicted_labels(self):
        file = os.path.join(self.directory, 'y_pred.npy')
        return np.load(file)
