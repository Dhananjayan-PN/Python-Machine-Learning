import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


df = pd.read_csv('Real estate.csv')
msk = np.random.rand(len(df)) < 0.9
train = df[msk]
test = df[~msk]
train_x, train_y, test_x, test_y = np.asanyarray(train[['X3 distance to the nearest MRT station']]), np.asanyarray(
    train[['Y house price of unit area']]), np.asanyarray(test[['X3 distance to the nearest MRT station']]), np.asanyarray(test[['Y house price of unit area']])


'''plt.scatter(df[['X3 distance to the nearest MRT station']],
            df[['Y house price of unit area']], color='blue')
plt.xlabel('Dist. to MST Station')
plt.ylabel('Cost per Unit Area')
plt.show()'''

regression = linear_model.LinearRegression()
regression.fit(train_x, train_y)
print('Coeff:', regression.coef_)
print('Intercept', regression.intercept_)

prediction = regression.predict(test_x)
print('Mean Absolute Error: ', np.mean(np.absolute(prediction - test_y)))
print('Mean Squared Error: ', np.mean((prediction - test_y)**2))
print('Root Mean Squared Error:', np.mean((prediction - test_y)**2)**0.5)
print('R2 Score:', r2_score(y_true=test_y, y_pred=prediction))

'''
df = pd.read_csv("FuelConsumptionCo2.csv")
newdf = df[['ENGINESIZE', 'TRANSMISSION',
            'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

msk = np.random.rand(len(newdf)) < 0.8
train = newdf[msk]
test = newdf[~msk]
train_x, train_y, text_x, test_y = np.asanyarray(train[['ENGINESIZE']]), np.asanyarray(
    train[['CO2EMISSIONS']]), np.asanyarray(test[['ENGINESIZE']]),  np.asanyarray(test[['CO2EMISSIONS']])
linear_reg = linear_model.LinearRegression()
linear_reg.fit(train_x, train_y)
print('coef:', linear_reg.coef_)
print('intercept:', linear_reg.intercept_)


test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = linear_reg.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" %
      np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
'''
