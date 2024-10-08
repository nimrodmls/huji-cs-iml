import numpy as np
import pandas as pd
import itertools
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from sklearn import metrics

import plotly.graph_objects as go
from cross_validate import cross_validate
from loss_functions import misclassification_error

def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    all_values = []
    all_weights = []
    
    def callback(weights, val, **kwargs):
        all_values.append(val)
        all_weights.append(weights)

    return callback, all_values, all_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [('L1', L1), ('L2', L2)]
    all_values = {name: {eta: None for eta in etas} for name, _ in modules}

    for (name, module), lr in itertools.product(modules, etas):
        callback, values, weights = get_gd_state_recorder_callback()
        weights.append(init)
        
        gd = GradientDescent(learning_rate=FixedLR(lr), callback=callback)
        # Note that passing X,y is redundant, as L1/L2 are not based on samples
        gd.fit(module(weights=np.copy(init)), None, None)
        all_values[name][lr] = values
        print(f"Module={name}, LR={lr}, Minimal Loss={np.min(values):.4f}")

        # Plotting the descent path for each Module-LR combination
        fig = plot_descent_path(module, np.array(weights), title=f'module={name}, lr={lr}')
        fig.write_image(f'{name}_fixed_{lr}.pdf')

    # Plotting convergence rate of each module-LR combination, one plot for each module
    for name, lr_values in all_values.items():
        fig = go.Figure(layout=go.Layout(title=f"{name} - Convergence Rate",
                                         xaxis=dict(title='GD Iteration'),
                                         yaxis=dict(title='Values')))
        
        for eta, values in lr_values.items():
            fig.add_trace(go.Scatter(x=list(range(len(values))), 
                                     y=values, 
                                     mode='lines+markers', 
                                     name=f'lr={eta}'))

        fig.write_image(f'{name}_fixed_conv_rate.pdf')

def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_train)
    fpr, tpr, thr = metrics.roc_curve(y_train, y_pred)
    arg_optim_alpha = np.argmax(tpr - fpr)
    optim_alpha = thr[arg_optim_alpha]
    fig = go.Figure(layout=go.Layout(title=f"Logistic Regression: ROC Curve",
                                         xaxis=dict(title='False-Positive Rate'),
                                         yaxis=dict(title='True-Positive Rate')))
    # Plotting the ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='red'), showlegend=False))
    # Plotting the perspective axis
    fig.add_trace(go.Scatter(x=(0, 1), y=(0, 1), mode='lines', line=dict(color='black'), showlegend=False))
    fig.write_image(f'logistic_regression_roc_curve.pdf')
    
    # Calculating the loss on the optimal alpha value
    model = LogisticRegression(alpha=optim_alpha)
    model.fit(X_train, y_train)
    print(f'Optimal α: {optim_alpha:.4f}, Loss: {model.loss(X_test, y_test):.4f}')

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambda_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    losses = np.zeros(shape=(len(lambda_values), 2))

    # Computing train & validation losses with cross-validation, on L1 only
    for i, lam in enumerate(lambda_values):
        gd = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
        model = LogisticRegression(penalty='l1', alpha=0.5, lam=lam, solver=gd)
        losses[i] = cross_validate(model, X_train, y_train, misclassification_error)
        print(f"Logistic Regression with L1 Regularization λ={lam} - Train Loss: {losses[i, 0]:.4f}, Validation Loss: {losses[i, 1]:.4f}")

    # Selecting the optimal lambda according to the validation loss
    opt_lambda = lambda_values[np.argmin(losses[:, 1])]
    gd = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(penalty='l1', alpha=0.5, lam=opt_lambda, solver=gd)
    model.fit(X_train, y_train)
    print(f"Optimal lambda: {opt_lambda}, Validation Loss: {np.min(losses[:, 1]):.4f}, Test Loss: {model.loss(X_test, y_test):.4f}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
