import matplotlib.pyplot as plt
import os


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

