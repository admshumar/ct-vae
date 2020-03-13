from tensorflow.keras.datasets import mnist
from time import time
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import os


class Rotator():
    @classmethod
    def view_image(cls, image):
        plt.imshow(image)
        plt.show()

    def __init__(self,
                 images,
                 labels,
                 number_of_rotations=2,
                 angle_of_rotation=30,
                 partition='train',
                 angle_threshold=180):
        """
        A rotator comes equipped with a data set, a directory to which augmented data sets are written, and augmentation
        parameters that determine the size and character of the augmented data set.
        :param images:
        :param directory:
        :param number_of_rotations:
        :param angle_of_rotation:
        """
        directory = os.path.abspath(os.path.join(os.getcwd(), 'data/mnist', partition))
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory

        if images is None:
            print("No data specified. Loading MNIST training set.")
            (x, y), (_, _) = mnist.load_data()
            self.images = x
            self.labels = y
            self.data_length = len(x)
        else:
            self.images = images
            self.labels = labels
            self.data_length = len(images)
        self.number_of_rotations = number_of_rotations
        self.angle_of_rotation = min(angle_of_rotation, 180)
        self.angle_threshold = angle_threshold
        self.angle_set = tuple(
            theta for theta in range(0, (number_of_rotations + 1) * angle_of_rotation, angle_of_rotation)
            if theta <= angle_threshold or theta >= 360 - angle_threshold)

    def get_rotated_images(self):
        print(f"Augmenting data set of shape {self.images.shape}.")
        t0 = time()
        augmentation_tuple = tuple(ndimage.rotate(self.images, theta, axes=(2, 1), reshape=False)
                                   for theta in self.angle_set)
        t1 = time()
        t = t1 - t0
        print(f"Completed in {t} seconds.")
        return np.concatenate(augmentation_tuple)

    def get_augmented_labels(self):
        return np.tile(self.labels, len(self.angle_set))

    def get_angular_labels(self):
        angular_array_tuple = tuple(np.asarray([angle] * len(self.images)) for angle in self.angle_set)
        return np.concatenate(angular_array_tuple)

    def save_rotated_images(self, list_of_digits):
        images_filename = f'digits={str(list_of_digits)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}'
        if self.angle_threshold < 180:
            images_filename += f'_threshold={self.angle_threshold}'
        images_filename += '.npy'
        images_filepath = os.path.abspath(os.path.join(self.directory, images_filename))
        augmented_data = self.get_rotated_images()
        np.save(images_filepath, augmented_data)

    def save_augmented_labels(self, list_of_digits):
        labels_filename = f'labels={str(list_of_digits)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}'
        if self.angle_threshold < 180:
            labels_filename += f'_threshold={self.angle_threshold}'
        labels_filename += '.npy'
        labels_filepath = os.path.abspath(os.path.join(self.directory, labels_filename))
        augmented_labels = self.get_augmented_labels()
        np.save(labels_filepath, augmented_labels)

    def save_angular_labels(self, list_of_digits):
        angles_filename = f'angles={str(list_of_digits)}_n_rot={self.number_of_rotations}_angle={self.angle_of_rotation}'
        if self.angle_threshold < 180:
            angles_filename += f'_threshold={self.angle_threshold}'
        angles_filename += '.npy'
        angles_filepath = os.path.abspath(os.path.join(self.directory, angles_filename))
        augmented_angles = self.get_angular_labels()
        np.save(angles_filepath, augmented_angles)

    def save_all(self, list_of_digits):
        self.save_rotated_images(list_of_digits)
        self.save_augmented_labels(list_of_digits)
        self.save_angular_labels(list_of_digits)

    def demo(self):
        digit_array = []
        for i in (4, 13, 18, 19):
            digit_array.append(self.images[i])
        digit_array = np.asarray(digit_array)

        rot = Rotator(digit_array)
        data = rot.get_rotated_images()

        for i in range(len(data)):
            self.view_image(data[i])


class MNISTRotator(Rotator):
    def __init__(self, list_of_digits, number_of_rotations, angle_of_rotation, partition='train', angle_threshold=180):
        if partition == 'test':
            print("Using MNIST test data.")
            (_, _), (x, y) = mnist.load_data()
        else:
            print("Using MNIST training data.")
            (x, y), (_, _) = mnist.load_data()
        class_indices = np.where(np.isin(y, list_of_digits))
        data = x[class_indices]
        labels = y[class_indices]
        self.list_of_digits = list_of_digits
        super(MNISTRotator, self).__init__(data, labels,
                                           number_of_rotations=number_of_rotations,
                                           angle_of_rotation=angle_of_rotation,
                                           partition=partition,
                                           angle_threshold=angle_threshold)

    def augment(self):
        self.save_all(self.list_of_digits)


"""
classes = list(range(10))
for c in classes:
    for dir in ['train', 'test']:
        rotator = MNISTRotator([c], 11, 30, partition=dir, angle_threshold=90)
        rotator.save_angular_labels(rotator.list_of_digits)
"""

