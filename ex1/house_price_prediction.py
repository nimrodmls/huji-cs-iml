import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import LinearRegression

TEST_SIZE = 0.25

def general_preprocess(X: pd.DataFrame, y: pd.Series = None):
    # Adding the room to lot ratio as a new feature, the smaller, the denser the lot is
    room_lot_ratio = np.where((X.bathrooms + X.bedrooms) > 0, X.sqft_lot / (X.bathrooms + X.bedrooms), 0)
    X.insert(len(X.columns), 'room_lot_ratio', room_lot_ratio)

    # Adding the residential units as a new feature, more units should reflect a higher price
    # a residential unit is defined as a single bathroom to multiple bedrooms
    residential_units = np.where(X.bathrooms > 0, X.bedrooms / X.bathrooms, 0)
    X.insert(len(X.columns), 'residential_units', residential_units)

    # Dropping the ID, sqft_lot15, sqft_living15 columns - not useful for prediction
    # Dropping the sqft_lot column as it's linearly dependent on sqft_above and sqft_basement
    X = X.drop(columns=['date', 'id', 'sqft_lot', 'sqft_lot15', 'sqft_living15'], axis=1)
    # Converting longitudes to positive values
    X.long = (X.long + 360.0) % 360.0

    # Making sure the year of renovation is not before the year of building
    # and if so, setting the year of renovation to the year of building
    X.loc[X.yr_renovated < X.yr_built, 'yr_renovated'] = X.loc[X.yr_renovated < X.yr_built, 'yr_built']

    if y is not None:
        # Filling missing prices with 0
        y.loc[pd.isnull] = 0

    return X

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X = general_preprocess(X, y)
    
    # Making sure that the square footage of living is not more than the square footage of the lot (lot = basement + above)
    # and if so, removing this sample as it's corrupted
    lot_mask = X.sqft_living <= X.sqft_above + X.sqft_basement
    X = X.loc[lot_mask, :]
    y = y.loc[lot_mask]
    if lot_mask.sum() != len(lot_mask):
        print('Removed corrupted samples due to sqft_living <= sqft_above + sqft_basement')

    # Removing samples with 0 bedrooms or 0 bathrooms
    bed_mask = X.bedrooms > 0
    bath_mask = X.bathrooms > 0
    mask = bed_mask & bath_mask
    X = X.loc[mask, :]
    y = y.loc[mask]
    if mask.sum() != len(mask):
        print('Removed samples with 0 bedrooms or 0 bathrooms')

    return X, y

def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    return general_preprocess(X)

def pearson_correlation(x: pd.Series, y: pd.Series):
    """
    Calculate Pearson Correlation between each feature and the response.
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    Returns
    -------
    corr : Pearson Correlation between each feature and the response (n_features, )
    """
    return x.cov(y) / (x.std() * y.std())

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> None:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.keys():
        plt.figure()
        plt.xlabel(feature)
        plt.ylabel('Response')
        plt.scatter(X[feature].values, y.values)
        plt.title(f'{feature} - Pearson Correlation: {pearson_correlation(X[feature], y)}')
        plt.savefig(os.path.join(output_path, f'{feature}.png'))
        plt.close()

if __name__ == '__main__':
    np.random.seed(0)

    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    
    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train) # TODO: REMOVE COMMENT

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    losses_over_p = {p: [] for p in range(10, 101)}
    for p in losses_over_p.keys():
        print(f'Testing model over {p}% of the training data')
        for _ in tqdm(range(10)):
            model = LinearRegression(include_intercept=True)
            trainset = X_train.sample(frac=p/100, replace=False)
            labels = y_train.loc[trainset.index]
            model.fit(trainset.to_numpy(), labels.to_numpy())
            losses_over_p[p].append(model.loss(X_test.to_numpy(), y_test.to_numpy()))

    plt.figure()
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss as function of Training Size')
    plt.plot(losses_over_p.keys(), [np.mean(losses) for losses in losses_over_p.values()], label='Mean Loss', color='blue')
    # Plotting the error ribbons
    for p, losses in losses_over_p.items():
        plt.errorbar(p, np.mean(losses), yerr=2*np.std(losses), fmt='bo--', ecolor='grey', capsize=5)
    plt.legend()
    plt.savefig('mean_loss_over_training_size.png')
