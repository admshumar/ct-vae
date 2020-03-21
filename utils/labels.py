import numpy as np
from scipy.stats import multivariate_normal


class OneHotEncoder:
    """
    Takes in a list of labels and returns a NumPy array of one-hot encodings.
    """

    def __init__(self, labels):
        classes = np.unique(labels)
        self.labels = labels
        self.classes = classes
        self.number_of_samples = len(labels)
        self.number_of_classes = len(classes)
        self.distribution_array = np.zeros(shape=(len(labels), len(classes)))

    def encode(self):
        """
        Fill self.distribution array with one-hot vectors corresponding to the classes indicated by self.labels.
        :return: A NumPy array of one-hot vectors.
        """
        one_hot_encoding = self.distribution_array
        for sample_index in range(self.number_of_samples):
            label = self.labels[sample_index]
            class_index = np.where(self.classes == label)
            one_hot_encoding[sample_index, class_index] += 1
        return one_hot_encoding

    def decode(self, one_hot_encoding, class_index):
        """
        Take in a set of one-hot encoded predictions and return one-versus rest binary predictions for a
        specified class.
        :param one_hot_encoding: A NumPy array of one-hot encoded vectors.
        :param class_index: A integer indicating the column corresponding to the set of binary predictions for a class.
        :return: A NumPy array of binary predictions.
        """
        return one_hot_encoding[:, class_index]

class OneHotDecoder:
    """
    Takes in a list of one-hot encoded vectors and return a class vector.
    """

    def __init__(self, labels):
        self.labels = labels
        self.number_of_samples = len(labels)

    def decode_to_multiclass(self):
        return np.asarray([np.where(self.labels[i] == 1)[0][0]
                           for i in range(self.number_of_samples)])

    def decode_to_binary(self, negative_class_index):
        return self.labels[:, negative_class_index]


class Flipper:
    """
    Alter the labels of a given list of labels based on a given list of angles.
    """

    def __init__(self, labels, angles, threshold=90):
        self.unique_labels = np.unique(labels)
        self.angles = angles
        self.labels = labels
        self.threshold = threshold

    def confuse_labels(self, idx):
        """
        Given an integer label, change it to any label other than the given label.
        :param idx: An integer index for both an array of angles and an array of labels.
        :return: An integer.
        """
        is_over_threshold = self.threshold <= self.angles[idx] <= 360 - self.threshold
        is_on_threshold = self.angles[idx] == self.threshold or self.angles[idx] == 360 - self.threshold
        confusing_labels = np.delete(self.unique_labels, np.where(self.unique_labels == self.labels[idx]))
        if not is_over_threshold:
            return self.labels[idx]
        elif is_on_threshold:
            return int(np.random.choice(self.unique_labels, 1))
        else:
            return int(np.random.choice(confusing_labels, 1))

    def change_labels(self):
        """
        Given an array of (sparse) labels, change a label to any other label, provided that the label in question
        corresponds to an angle in a specified interval.
        :return: A NumPy array of confused labels.
        """
        print('Changing labels.')
        new_labels = [self.confuse_labels(idx) for idx in range(len(self.labels))]
        return np.asarray(new_labels)


class Smoother(OneHotEncoder):
    """
    Perform label smoothing when given a list of labels for a data set.
    """

    def __init__(self, labels, alpha=0.1, smooth_labels=None):
        super(Smoother, self).__init__(labels)
        self.alpha = float(alpha)
        self.smooth_labels = smooth_labels

    def smooth(self):
        true_distribution = (1 - self.alpha) * self.encode()
        if self.smooth_labels is not None:
            smoothing_distribution = self.alpha * self.smooth_labels
        else:
            smoothing_distribution = (self.alpha / self.number_of_classes) * self.distribution_array
        return np.add(true_distribution, smoothing_distribution)


class GaussianSoftLabels:
    """
    Uses a Gaussian mixture model to construct soft labels for data points in Euclidean space.
    """

    def __init__(self, data, means, covariances, labels=None):
        self.data = data
        self.means = means
        self.covariances = covariances
        self.labels = labels

    def get_class_distribution(self, point):
        class_densities = [multivariate_normal.pdf(point,
                                                   self.means[j],
                                                   self.covariances[j]) for j in range(len(self.means))]
        class_distribution = np.asarray(class_densities)
        class_distribution *= 1 / np.sum(class_distribution)
        return np.asarray([class_distribution])

    def get_soft_labels(self):
        class_distribution_tuple = tuple(self.get_class_distribution(self.data[i]) for i in range(len(self.data)))
        soft_labels = np.concatenate(class_distribution_tuple)
        return soft_labels

    def smooth_gaussian_mixture(self, alpha=0.1):
        if self.labels:
            soft_labels = self.get_soft_labels()
            assert self.labels.shape == soft_labels, "Labels have incompatible shapes."
            true_distribution = (1 - alpha) * self.labels
            soft_distribution = alpha * soft_labels
            return true_distribution + soft_distribution
        else:
            print('No labels provided!')
            return None
