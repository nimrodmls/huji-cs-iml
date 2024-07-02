from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from polynomial_fitting import PolynomialFitting

TEST_RATIO = 0.25

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
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.insert(len(df.columns), 'DayOfYear', [row.Date.dayofyear for i, row in df.iterrows()])

    # Some samples have the weird temperature value of -72.77777777777777
    # We will remove these samples from the dataset
    df = df.loc[df.Temp != -72.77777777777777]

    return df

if __name__ == '__main__':
    # Setting random seed for reproducibility
    np.random.seed(42)

    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_data = df.loc[df.Country == 'Israel']

    # Q3.1 - Plotting daily temperatures
    plt.figure()
    plt.title('Daily Temperature in Israel')
    plt.xlabel('Day Of Year')
    plt.ylabel('Temperature')
    for year in israel_data.Year.unique():
        data_per_year = israel_data.loc[israel_data.Year == year]
        plt.scatter(data_per_year.DayOfYear, data_per_year.Temp, s=3, label=f'{year}')
    plt.legend(fontsize='small', title='Year')
    plt.savefig('daily_temperature.pdf')
    plt.close()

    # Q3.2 - Plotting temperature standard deviation per month
    israel_data.groupby('Month').Temp.std().plot(
        ylabel='STD', kind='bar', title='Temperature Standard Deviation per Month')
    plt.savefig('temp_std_per_month.pdf')
    plt.close()

    # Question 4 - Exploring differences between countries
    country_data = df.groupby(['Country', 'Month'])

    plt.figure()
    plt.xlabel('Month')
    plt.ylabel('Mean Temperature')
    countries_mean_temp = country_data.Temp.mean()
    countries_std_temp = country_data.Temp.std()
    for country in df.Country.unique():
        current_country_temps = countries_mean_temp[country, :]
        current_country_std = countries_std_temp[country, :]
        plt.errorbar(current_country_temps.keys(), current_country_temps, yerr=current_country_std, capsize=5, label=country)
    plt.legend()
    plt.savefig('mean_temp_per_month.pdf')
    plt.close()

    # Question 5 - Fitting model for different values of `k`
    X = israel_data.DayOfYear.to_numpy()
    y = israel_data.Temp.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=0)

    max_k = 10
    losses_over_k = { k: 0 for k in range(1, max_k + 1) }
    for k in losses_over_k.keys():
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        losses_over_k[k] = np.around(model.loss(X_test, y_test), 2)

    print('Losses over k: ', losses_over_k)
    plt.figure()
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Loss')
    plt.title('Loss as function of Polynomial Degree')
    plt.bar(losses_over_k.keys(), losses_over_k.values(), label='Loss', color='blue')
    plt.savefig('loss_over_k.pdf')
    plt.close()

    # Question 6 - Evaluating fitted model on different countries
    chosen_k = 5
    model = PolynomialFitting(chosen_k)
    model.fit(X_train, y_train)

    non_israel_countries = df.Country.unique()['Israel' != df.Country.unique()]
    losses_over_country = { country: 0 for country in non_israel_countries }
    # Iterating on all non-israel countries
    for country in losses_over_country.keys():
        country_data = df.loc[df.Country == country]
        X_country = country_data.DayOfYear.to_numpy()
        y_country = country_data.Temp.to_numpy()
        losses_over_country[country] = np.around(model.loss(X_country, y_country), 2)
    
    plt.figure()
    plt.xlabel('Country')
    plt.ylabel('Loss')
    plt.title('Loss as function of Country')
    plt.bar(losses_over_country.keys(), losses_over_country.values(), label='Loss', color='blue')
    plt.savefig('loss_over_country.pdf')
    plt.close()