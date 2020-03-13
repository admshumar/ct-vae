import os
from utils import classifiers


class LatentSpaceClassifier:
    def __init__(self, data, labels, experiment_directory):
        self.data = data
        self.labels = labels
        self.experiment_directory = experiment_directory
        file_path = os.path.abspath(os.path.join(experiment_directory, 'classifiers.txt'))
        self.file_path = file_path
        self.file = open(file_path, "w+")
        
    def get_mixture_model(self):
        return classifiers.fit_mixture_model_on_latent_space(self.data, self.labels)

    def get_logistic_regression(self):
        return classifiers.logistically_regress_on_latent_space(self.data, self.labels)

    def get_support_vector_classification(self):
        return classifiers.sv_classify_on_latent_space(self.data, self.labels, kernel='rbf')

    def report_score(self, model, text):
        score = model.score(self.data, self.labels)
        score_string = text + f" {score}\n"
        print(text, score)
        self.file.write(score_string)

    def get_soft_labels(self, model, data):
        """
        From either a trained logistic regression model, a trained support vector classifier model, or a Gaussian
        mixture model, return the class distributions that are found by these models.
        :param model: A trained scikit-learn classifier model.
        :return: A NumPy array of floats corresponding to class distributions.
        """
        return model.predict_proba(data)

    def report(self, with_svc=False):
        logistic_regression = self.get_logistic_regression()
        self.report_score(logistic_regression, "Logistic regression model score:")

        mixture_model = self.get_mixture_model()
        self.report_score(mixture_model, "Gaussian mixture model per-sample average log-likelihood:")

        if with_svc:
            svc = self.get_support_vector_classification()
            self.report_score(svc, "Support vector classifier mean accuracy:")

        self.file.close()
