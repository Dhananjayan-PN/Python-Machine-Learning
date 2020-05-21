import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('Regression\china_gdp.csv')
x_data, y_data = (df["Year"].values, df["Value"].values)
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)
'''plt.plot(df[['Year']], df[['Value']], color='blue')
plt.show()'''

# We're using a logistic function as it represents the trends in the data well


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


beta_1 = 0.10
beta_2 = 1990.0

Y_pred = sigmoid(x_data, beta_1, beta_2)

# plot initial prediction against datapoints
'''plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()'''

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print('beta1 = {} beta2 = {}'.format(popt[0], popt[1]))

# Visualize the fit/model
y = sigmoid(xdata, popt[0], popt[1])
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(xdata, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
