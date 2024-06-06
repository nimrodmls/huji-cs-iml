from classifiers import Perceptron, LDA, GaussianNaiveBayes
from loss_functions import accuracy
from typing import Tuple, List
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), 
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        max_iter = 1000
        losses = []
        # Creating a simple perceptron object with a callback for recording losses
        # in each fitting iteration
        perceptron = Perceptron(
            max_iter=max_iter,
            callback=lambda perceptron, X, y: losses.append(perceptron.loss(X, y)))
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label="Training Loss", color="red")
        plt.title(f"Perceptron Training Loss - {n}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb_model = GaussianNaiveBayes()
        gnb_model.fit(X, y)

        lda_model = LDA()
        lda_model.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Gaussian Naive Bayes", "LDA"))
        figure.update_layout(title=f"Classifier Comparison - Dataset: {f}")
        figure.add_trace(go.Scatter(x=X[y == 0, 0], y=X[y == 0, 1], mode="markers", marker_color="blue", name="Class 0"),
                            row=1, col=1)
        figure.add_trace(go.Scatter(x=X[y == 1, 0], y=X[y == 1, 1], mode="markers", marker_color="red", name="Class 1"),
                            row=1, col=1)
        

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    #compare_gaussian_classifiers()
