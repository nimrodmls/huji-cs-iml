import numpy as np
import pandas as pd
from polynomial_fitting import PolynomialFitting

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    pass


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    model = PolynomialFitting(3)
    model.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]), np.array([[1], [2], [3]]))


    # Question 3 - Exploring data for specific country

    # Question 4 - Exploring differences between countries

    # Question 5 - Fitting model for different values of `k`

    # Question 6 - Evaluating fitted model on different countries

    pass
