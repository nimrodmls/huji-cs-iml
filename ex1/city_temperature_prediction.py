import numpy as np
import matplotlib.pyplot as plt
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
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.insert(len(df.columns), 'DayOfYear', [row.Date.dayofyear for i, row in df.iterrows()])

    df = df.loc[df.Temp != -72.77777777777777]
    return df

if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    #model = PolynomialFitting(3)
    #model.fit(np.array([[1,2,3], [4,5,6], [7,8,9]]), np.array([[1], [2], [3]]))

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
    plt.legend()
    plt.show()

    # Q3.2 - Plotting temperature standard deviation per month
    israel_data.groupby('Month').Temp.std().plot(
        ylabel='STD', kind='bar', title='Temperature Standard Deviation per Month')
    plt.show()

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
        #plt.plot(current_country_temps.keys(), current_country_temps, label=country)
        plt.errorbar(current_country_temps.keys(), current_country_temps, yerr=current_country_std, capsize=5, label=country)
    plt.legend()
    plt.show()

    # Question 5 - Fitting model for different values of `k`

    # Question 6 - Evaluating fitted model on different countries

    pass
