# Sample-Linear-Regression
Sample-Linear-Regression
![image.png](attachment:image.png)

# Understanding the Data

![image.png](attachment:image.png)

#import Libraries
import numpy as np
import pandas as pd

#Load Dataset
advertising = pd.read_csv("D:\\DATA SCIENCE\\ML Projects\\Simple-Linear-Regression\\TV-MARKETING\\tvmarketing.csv")

Now, let's check the structure of the advertising dataset.

#Display the first five rows
advertising.head()

# Display the last 5 rows
advertising.tail()

# Let's check the columns
advertising.info()

# Check the shape of the DataFrame (rows, columns)
advertising.shape

# Let's look at some statistical information about the dataframe.
advertising.describe()

# Visualising Data Using Plot

# Visualise the relationship between the features and the response using scatterplots
advertising.plot(x='TV',y='Sales',kind='scatter')

# Perfroming Simple Linear Regression

![image.png](attachment:image.png)

# Generic Steps in Model Building using sklearn

Before you read further, it is good to understand the generic structure of modeling using the scikit-learn library. Broadly, the steps to build any model can be divided as follows:

# Preparing X and y

![image.png](attachment:image.png)

# Putting feature variable to X
X = advertising['TV']

# Print the first 5 rows
X.head()

# Putting response variable to y
y = advertising['Sales']

# Print the first 5 rows
y.head()

# Splitting Data into Training and Testing Sets

#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=0000)

print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))

#Press Tab+Shift to read the documentation
train_test_split      


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#It is a general convention in scikit-learn that observations are rows, while features are columns. 
#This is needed only when you are using a single feature; in this case, 'TV'.

import numpy as np
#Simply put, numpy.newaxis is used to increase the dimension of the existing array by one more dimension,
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Performing Linear Regression

# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# Fit the model using lr.fit()
lr.fit(X_train, y_train)

# Coefficients Calculation

# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)

y=6.989+0.0464×TV
 

Now, let's use this equation to predict our sales.

# Predictions

# Making predictions on the testing set
y_pred = lr.predict(X_test)

type(y_pred)

# Computing RMSE and R^2 Values

RMSE is the standard deviation of the errors which occur when a prediction is made on a dataset. This is the same as MSE (Mean Squared Error) but the root of the value is considered while determining the accuracy of the model

y_test.shape # cheek the shape to generate the index for plot

# Actual vs Predicted
import matplotlib.pyplot as plt
c = [i for i in range(1,61,1)]         # generating index 
fig = plt.figure()
plt.plot(c,y_test, color="blue", linewidth=2, linestyle="-")
plt.plot(c,y_pred, color="red",  linewidth=2, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Sales', fontsize=16)                       # Y-label

# Error terms
c = [i for i in range(1,61,1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# this mse =7.9 means that this model is not able to match the 7.9 percent of the values
# r2 means that your model is 72% is accurate on test data .


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,c='blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.grid()

