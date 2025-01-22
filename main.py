#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset and some lookovers on it
cost_df = pd.read_csv("EconomiesOfScale.csv")
cost_df.head(100)
cost_df.describe()
cost_df.info()

#Visualising the data
sns.jointplot(x='Number of Units', y='Manufacturing Cost', data = cost_df)

sns.lmplot(x='Number of Units', y='Manufacturing Cost', data=cost_df)


#Creating training dataset
X = cost_df[['Number of Units']]
y = cost_df['Manufacturing Cost']

# Note that we used the entire dataset for training only
X_train = X
y_train = y

#SOLUTION: LINEAR ASSUMPTION
#Model Training
y_train.shape
X_train.shape

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept =True)

regressor.fit(X_train,y_train)

print('Model Coefficient: ', regressor.coef_)


#Visualizing the result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.ylabel('Cost Per Unit Sold [dollars]')
plt.xlabel('Number of Units [in Millions]')
plt.title('Unit Cost vs. Number of Units [in Millions](Training dataset)')


#SOLUTION 2: Polynomial Assumption

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
# import a class and instantiate an object from that class

# Transform the matrix of features X into a multi array of features X_Columns
# which contains the original features and their associated polynomial terms
X_columns = poly_regressor.fit_transform(X_train)

print(X_columns)

regressor = LinearRegression()
regressor.fit(X_columns, y_train)


print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

#Visualize the result
X_train.shape

y_predict = regressor.predict(poly_regressor.fit_transform(X_train))

y_predict.shape

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_predict, color = 'blue')
plt.ylabel('Cost Per Unit Sold [dollars]')
plt.xlabel('Number of Units [in Millions]')
plt.title('Unit Cost vs. Number of Units [in Millions](Training dataset)')

#Now we can observe that the bestfit line is on the trend with the plot



