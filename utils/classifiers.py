import numpy as np
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from utils.labels import OneHotDecoder


def logistically_regress_on_latent_space(data, labels):
    if len(labels.shape) > 1:
        labels = OneHotDecoder(labels).decode_to_multiclass()
    logistic_regressor = LogisticRegression(max_iter=1000)
    logistic_regressor.fit(data, labels)
    return logistic_regressor


def fit_mixture_model_on_latent_space(data, labels):
    unique_labels = np.unique(labels)
    number_of_classes = len(unique_labels)
    mixture_model = GaussianMixture(n_components=number_of_classes, n_init=10)
    mixture_model.fit(data, labels)
    return mixture_model


def sv_classify_on_latent_space(data, labels, kernel='linear'):
    support_vector_classifier = svm.SVC(kernel=kernel, probability=True)
    support_vector_classifier.fit(data, labels)
    return support_vector_classifier
