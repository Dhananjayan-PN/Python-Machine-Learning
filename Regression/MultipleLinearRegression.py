import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Regression\FuelConsumptionCo2.csv")

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train_x, train_y, test_x, test_y = (
    np.asanyarray(train[[
        "ENGINESIZE",
        "CYLINDERS",
        "FUELCONSUMPTION_COMB",
        "FUELCONSUMPTION_COMB_MPG",
    ]]),
    np.asanyarray(train[["CO2EMISSIONS"]]),
    np.asanyarray(test[[
        "ENGINESIZE",
        "CYLINDERS",
        "FUELCONSUMPTION_COMB",
        "FUELCONSUMPTION_COMB_MPG",
    ]]),
    np.asanyarray(test[["CO2EMISSIONS"]]),
)

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

predicted = regr.predict(test_x)
print("Coeff:", regr.coef_)
print("Intercept:", regr.intercept_)

print("MSE:", np.mean(predicted - test_y)**2)
print("Variance Score:", regr.score(test_x, test_y))
"""plt.scatter(train.CYLINDERS, train.CO2EMISSIONS)
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.show()"""
