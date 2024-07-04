import numpy as np
from typing import Tuple
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_model = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost_model.fit(train_X, train_y)
    train_errors = [adaboost_model.partial_loss(train_X, train_y, learner_count) 
                    for learner_count in range(n_learners + 1)]
    test_errors = [adaboost_model.partial_loss(test_X, test_y, learner_count)
                   for learner_count in range(n_learners + 1)]

    plt.figure()
    plt.plot(train_errors, label='Train error')
    plt.plot(test_errors, label='Test error')
    plt.legend()
    plt.xlabel('Number of learners')
    plt.ylabel('Misclassification loss')
    plt.title(f'Train & Test loss on AdaBoost with {n_learners} learners')
    plt.savefig(f"q1_train_test_loss_{noise}.pdf")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    figure = make_subplots(rows=1, cols=len(T), subplot_titles=[f'T = {t}' for t in T])
    figure.update_layout(title='Decision surfaces of AdaBoost with varied number of learners')
    for i, t in enumerate(T):
        predict = lambda X: adaboost_model.partial_predict(X, t)
        figure.add_trace(
            decision_surface(predict, lims[0], lims[1], showscale=False), 
            row=1, col=i+1)
        figure.add_trace(
            go.Scatter(x=test_X[:, 0], 
                       y=test_X[:, 1], 
                       mode='markers', 
                       marker=dict(color=test_y, colorscale=class_colors(2)), 
                       showlegend=False), 
            row=1, col=i+1)
    figure.write_image(f"q2_t_decision_surfaces_{noise}.pdf")

    # Question 3: Decision surface of best performing ensemble
    accuracies = [(adaboost_model.partial_predict(test_X, t) == test_y).sum() / test_size
                  for t in range(n_learners + 1)]
    best_ensemble = np.argmax(accuracies)
    predict = lambda X: adaboost_model.partial_predict(X, t)
    figure = make_subplots(rows=1, cols=1)
    figure.update_layout(
        title=f'Decision surfaces of AdaBoost - Best Accuracy // Accuracy: {accuracies[best_ensemble]:.2f}, T = {best_ensemble}')
    figure.add_trace(
        decision_surface(predict, lims[0], lims[1], showscale=False), 
        row=1, col=1)
    figure.add_trace(
        go.Scatter(x=test_X[:, 0], 
                   y=test_X[:, 1], 
                   mode='markers', 
                   marker=dict(color=test_y, colorscale=class_colors(2)), 
                   showlegend=False), 
        row=1, col=1)
    figure.write_image(f"q3_best_{noise}.pdf")

    # Question 4: Decision surface with weighted samples
    figure = make_subplots(rows=1, cols=1)
    figure.update_layout(
        title=f'Decision surfaces of AdaBoost - Weights at last iteration')
    figure.add_trace(
        decision_surface(adaboost_model.predict, lims[0], lims[1], showscale=False), 
        row=1, col=1)
    figure.add_trace(
        go.Scatter(x=train_X[:, 0], 
                   y=train_X[:, 1], 
                   mode='markers', 
                   marker=dict(color=train_y, 
                               colorscale=class_colors(2),
                               size=adaboost_model.D_[-1] / np.max(adaboost_model.D_[-1]) * 5), 
                   showlegend=False), 
        row=1, col=1)
    figure.write_image(f"q4_weighted_{noise}.pdf")

if __name__ == '__main__':
    np.random.seed(0)
    # Questions 1-4
    fit_and_evaluate_adaboost(noise=0)
    # Question 5
    fit_and_evaluate_adaboost(noise=0.4)