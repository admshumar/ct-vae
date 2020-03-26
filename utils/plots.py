import os
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

class LatentSpaceTSNE:
    def __init__(self, data, labels, directory, number_of_tnse_components=2):
        if number_of_tnse_components in (2, 3):
            self.number_of_tnse_components = number_of_tnse_components
        else:
            self.number_of_tnse_components = 2
        self.data = data
        self.directory = directory
        self.labels = np.argmax(labels, axis=-1)
        self.classes = np.unique(np.argmax(labels, axis=-1))
        self.perplexity_list = [10, 30, 50]
        color_list = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']
        self.color_list = color_list
        self.color_dict = {i: color_list[i] for i in range(len(color_list))}

    def get_embedding(self, perplexity):
        return TSNE(n_components=self.number_of_tnse_components,
                             perplexity=perplexity).fit_transform(self.data)

    def save_embedding(self, perplexity):
        embedded_data = self.get_embedding(perplexity)
        filename = 'latent_tsne_{}.npy'.format(perplexity)
        file_path = os.path.join(self.directory, filename)
        np.save(file_path, embedded_data)

    def save_tsne(self):
        for perplexity in self.perplexity_list:
            embedded_data = self.get_embedding(perplexity)
            fig_3d = plt.figure(dpi=200)

            if self.number_of_tnse_components == 2:
                ax = fig_3d.add_subplot()
                for label in self.classes:
                    data = embedded_data[np.where(self.labels == label)]
                    ax.scatter(data[:, 0],
                               data[:, 1],
                               c=self.color_dict[label],
                               label=str(label))
                ax.legend(loc='best')
            else:
                ax = fig_3d.add_subplot(projection='3d')
                for label in self.classes:
                    data = embedded_data[np.where(self.labels == label)]
                    ax.scatter(data[:, 0],
                               data[:, 1],
                               data[:, 2],
                               c=self.color_dict[label],
                               label=str(label))
                ax.legend(loc='best')

            ax.set_title('t-SNE with Perplexity {}'.format(perplexity))
            filename = 'latent_tsne_{}.png'.format(perplexity)
            file_path = os.path.join(self.directory, filename)
            fig_3d.savefig(file_path)


def plot(model_history, directory, function_list, filename, plot_title, metric, x_variable, location='upper right'):
    """
    :param model_history: A dictionary of loss values.
    :param directory: A string indicating the directory to which the image is written.
    :param functions: A set of strings indicating the functions to be plotted.
    :param filename: A string indicating a filename for the plot.
    :param plot_title: A string indicating the title of the plot.
    :param metric: A string indicating the label for the vertical axis.
    :param x_variable: A string indicating the label for the horizontal axis.
    :param location: A string indicating the location of the key in the plot.
    :return: None
    """
    filepath = os.path.join(directory, filename + '.png')
    fig = plt.figure(dpi=200)
    for function in function_list:
        plt.plot(model_history.history[function])
    plt.title(plot_title)
    plt.ylabel(metric)
    plt.xlabel(x_variable)
    plt.legend(['Train', 'Val'], loc=location)
    fig.savefig(filepath)
    plt.close(fig)


def loss(model_history, directory):
    plot(model_history, directory, ['loss', 'val_loss'], 'loss', 'Model Loss', 'Loss', 'Epochs')


def accuracy(model_history, directory):
    plot(model_history, directory, ['acc', 'val_acc'], 'accuracy',
         'Model Accuracy', 'Accuracy', 'Epochs', location='lower right')
