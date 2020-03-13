from models.vae_dense import DenseVAE
from models.latent_space_classifiers import LatentSpaceClassifier
from models.mnist_cnn_classifier import MNISTCNNClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

import numpy as np
import os

class Calibrator:
    def __init__(self, train=False):
        self.auto_encoder = auto_encoder
        self.classifier = classifier

    def calibrate(self):
        pass

    def train_classifier(self):
        pass
