# 0. import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 1. import Dataset

dataset = pd.read_csv('data/Position_Salaries.csv')

# 2. Split Dataseet into Features and Target . Train and Test .

X = dataset.iloc[: , 1:-1]
y = dataset.iloc[: , -1 ]

# 3. Split Dataseet into Train and Test .

from sklearn.linear_model import LinearRegression
linear_reg  = LinearRegression()
linear_reg.fit(X , y)
y_lin_pred = linear_reg.predict(X)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly , y)
y_lin_pred2 = linear_reg2.predict(X_poly)

# visualisation linear regrression result
plt.scatter(X,y)
plt.plot(X ,y_lin_pred , color='red')
plt.title('Truth (Linear Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

# visualisation polymoninal result
plt.scatter(X, y)
plt.plot( X , y_lin_pred2 , color='red')
plt.title('Truth (Polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()





