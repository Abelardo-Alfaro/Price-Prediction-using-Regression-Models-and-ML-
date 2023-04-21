import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


path = 'C:/Users/abeal/Desktop/work/Predicting Models/Predicting car prices/clean_df.csv'
df = pd.read_csv(path)
df.head()

######################################### CREATING MODELS ##############################################################

'''
We are now going to create a different models that will try to predict "Yhat". 
    - Linear Regression
    - Multi linear Regression 
    - Polynomical Regression 
'''
# Let's start with the LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

X = df[['horsepower']]
Y = df['price']

lm = LinearRegression()
lm.fit(X, Y)
Yhat = lm.predict(X)

# If we want to know the slope (m) and the intercept (n) we use: y = mx + n
n = lm.intercept_
m = lm.coef_
print(n, m)


# then with the MULTI LINEAR REGRESSION:

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']]
lm.fit(Z, Y)
Yhat2 = lm.predict(Z)

# If we want to know the slope (m) and the intercept (n) we use: y = mx + n
n2 = lm.intercept_
m2 = lm.coef_
print(n2, m2)

#### checking the LR and the MLR

# 1. let's visualize a residual plot to see if we need to do a nonlinear model (polynomial model).
# Analysing the residual plot we can see that the data isn't random around the X-axis and as such it might suggest
# that we need nonlinear model
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-L/100km'],y=df['price'])
plt.show()

# 2. Nos it's the MLR, to check the model we will need to create a distribution plot
# Analysing the distribution plot We can see that the fitted values are reasonably close to the actual values since the
# two distributions overlap a bit. However, there is definitely some room for improvement.
plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()

# Now it's time for the MULTI LINEAR REGRESSION (or POLYNOMIAL LINEAR REGRESSION)
x = df['highway-L/100km']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

# Another strategy could be making a PIPELINE to simply the steps of processing the data.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# We create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)

# First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of
# StandardScaler taking float inputs.
# Then, we can normalize the data, perform a transform and fit the model simultaneously.
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(ypipe[0:4])

########### MEASURE WHICH MODEL IS THE BEST ############################################################################



'''
We will use two methods that will help us determine which of 3 method is the best method for predicting our target. 
 1. R^2 (coefficient of determination) 
    - which is a measure to indicate how close the data is to the fitted regres. line
 2. Mean Squared Error (MSE)
    - Measures the average of the squares of errors. Difference between actual value (y) and the estimated value (ŷ). 
'''

# Let's calculate first the R^2 - the closer to 1 the better.

# for the LINEAR REGRESSION model:
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

# for the MULTI LINEAR REGRESSION model
lm.fit(Z, Y)
# Find the R^2
print('The R-square is: ', lm.score(Z, Y))

# for the POLYNOMIAL REGRESSION model
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)



# Let's calculate now the MSE - Smaller MSE is better.
from sklearn.metrics import mean_squared_error

# for the LINEAR REGRESSION model:
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# for the MULTI LINEAR REGRESSION model:
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Yhat2))

# for the POLYNOMIAL REGRESSION model:
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], p(x)))


################################### Ridge Regresson and Gris Search ####################################################

'''
Ridge Regression is a type of regularized linear regression that helps to prevent overfitting. we will review Ridge 
Regression and see how the parameter alpha changes the model. Just a note, here our test data will be used as validation
data.

We will also randomly split the data into training and testing and define the target
'''

# We will place the target data price in a separate dataframe y_data:
y_data = df['price']

#Drop price data in dataframe x_data:
x_data=df.drop('price',axis=1)

# Now, we randomly split our data into training and testing data using the function train_test_split.
from sklearn.model_selection import train_test_split

# 90% for training, 10% for data testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km','normalized-losses', 'symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km','normalized-losses', 'symboling']])

from sklearn.linear_model import Ridge

# Let's create a Ridge regression object, setting the regularization parameter (alpha) to 0.1
RigeModel=Ridge(alpha=0.1)

# similarly we can train the model with .fit() it and predict it by using the .predict()
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)

# let's see the result:
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

# We select the value of alpha that minimizes the test error. To do so, we can use a for loop. We have also created a
# progress bar to see how many iterations we have completed so far

from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0, 1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha)
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)

    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# Let's plot the progress bar and make conclusions:
# The red line represents the R^2 of the training data. As alpha increases the R^2 decreases.
# Therefore, as alpha increases, the model performs worse on the training data
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

# Now we are going to do a Grid Search

from sklearn.model_selection import GridSearchCV

# We create a dictionary of parameter values:
parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]

# Create a Ridge regression object:
RR = Ridge()

# Create a Ridge grid search object:
Grid1 = GridSearchCV(RR, parameters1,cv=4)

# we will fit the model:
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']], y_data)

# The object finds the best parameter values on the validation data. We can obtain the estimator with the best
# parameters and assign it to the variable BestRR as follows:
BestRR = Grid1.best_estimator_
print(BestRR)

# Now we test our model on the test data:
print(BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-L/100km']], y_test))










