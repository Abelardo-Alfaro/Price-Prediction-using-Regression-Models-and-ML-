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

######################################### Visualizing & Statiscal Stage  ###############################################

'''
In the Loading Phase we would need to analyze the individual features to be able to choose it to load it in the predictive 
model.
'''

'''
(1) Analyzing Individual Feature Patterns Using Visualization:

'''

# we will start by visualizing a correlation table of all the variables between each other with a .corr() method
# and visualizing it with a heatmap of correlation. but first we need to drop all the columns that are strings
corr1 = df.drop(labels=['Unnamed: 0','make', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type',
                        'num-of-cylinders', 'fuel-system', 'horsepower-binned',	'fuel-type-diesel',	'fuel-type-gas',
                        'fuel-type-diesel',	'fuel-type-gas'], axis=1, inplace=False).corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr1, annot=True, )
plt.show()

'''
Now that we can see the correlation in a heatmap matrix, we need to now differentiate between:

    (1.1) - Continuous Numerical Variables: int64, float64 type
        [] Regression plots
    (1.2) - Categorical Values: object type
        [] Boxplot
        [] Pivot Tables (with avg price)
        [] heatmap
    (1.3) - Determining the Correlation & Causation
        [] Pearson Correlation
        [] P-Value
        [] ANOVA Test (Analysis of Variance) 
'''

'''
  (1.1) - Continuous Numerical Variables: int64, float64 type
'''
# A great way to study the continuous variables is by using scatter plots with a fitted regression plot.

# scatter plot1: Engine size as potential predictor variable of price
# Analyzing the regression plot, line and the correlation result, we can say it seems that the engine size could
# be a very good way of predicting the price of a car. Correlation coefficient close to 1 (0.87).
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.show()
print(df[["engine-size", "price"]].corr())

# scatter plot2: Highway mile per gallon as potential predictor variable of price
# Analyzing the regression plot, line and the correlation result, we can say it seems that the engine size could
# be a very good way of predicting the price of a car. Correlation coefficient close to 1 (0.80).
sns.regplot(x="highway-L/100km", y="price", data=df)
plt.show()
print(df[['highway-L/100km', 'price']].corr())

# scatter plot3: width as potential predictor variable of price
# Analyzing the regression plot, line and the correlation result, we can say it seems that the engine size could
# be a very good way of predicting the price of a car. Correlation coefficient close to 1 (0.75).
sns.regplot(x="width", y="price", data=df)
plt.show()
print(df[['width', 'price']].corr())

# scatter plot4: curb-weight as potential predictor variable of price
# Analyzing the regression plot, line and the correlation result, we can say it seems that the engine size could
# be a very good way of predicting the price of a car. Correlation coefficient close to 1 (0.83).
sns.regplot(x="curb-weight", y="price", data=df)
plt.show()
print(df[["curb-weight", "price"]].corr())

# scatter plot5: horsepower as potential predictor variable of price
# Analyzing the regression plot, line and the correlation result, we can say it seems that the engine size could
# be a very good way of predicting the price of a car. Correlation coefficient close to 1 (0.81).
sns.regplot(x="horsepower", y="price", data=df)
plt.show()
print(df[["horsepower", "price"]].corr())

'''
  (1.2) - Categorical Variables: 
    These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories.
    The categorical variables can have the type "object" or "int64"
'''
'''
    [] Boxplot
'''
# we will use Boxplot to see the data distribution of the categorical variables with respect the price.
# boxplot 1: price vs body-style
# We see that the distributions of price between the different body-style categories have a significant overlap,
# so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price":
sns.boxplot(x="body-style", y="price", data=df)
plt.show()

# boxplot 2: price vs engine-location
# Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct
# enough to take engine-location as a potential good predictor of price
sns.boxplot(x="engine-location", y="price", data=df)
plt.show()

# boxplot 3: price vs engine-location
# Here we see that the distribution of price between the different drive-wheels categories differs.
# As such, drive-wheels could potentially be a predictor of price.
sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()

# Now that we know how the data is distributed, lets count the time each variable appears for each selected category
# Category 1: Drive wheels
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# Category 2: Drive wheels
drive_wheels_counts = df['engine-location'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'engine-location'
print(drive_wheels_counts)

'''
    [] Pivot Table
'''

# It's always useful to analyze the data in a way that it is easy to visualize while still taking into account two variables
# columns and rows, and one internal data that is calculated based on the necesity. For this we need a pivot table, and
# we will start with creating a table with the uninque(), groupby(), and mean() methods.

# grouping results
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
print(grouped_test1)
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)  # fill missing values with 0
print(grouped_pivot)


'''
[] Heatmap of the pivot table
'''
# Now to be able to visualize it better we will create a heatmap with the data of the pivot table.
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

'''
    (1.3) - Determining the Correlation & Causation:
    Correlation: a measure of the extent of interdependence between variables.

    Causation: the relationship between cause and effect between two variables.

    It is important to know the difference between these two. Correlation does not imply causation. 
    Determining correlation is much simpler the determining causation as causation may require independent experimentation.
'''

'''
    [] Pearson Correlation
'''

# We can use the .corr() method to see a matrix of correlation where.
# 1: Perfect Positive Linear Correlation
# 0: No linear correlation, the two variables most likely do not affect each other.
# -1: Perfect Negative Linear Correlation

correlation_matrix = corr1
print(correlation_matrix)

'''
    [] P-Value:
        What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.

        By convention, when the

            p-value is < 0.001: we say there is strong evidence that the correlation is significant.
            p-value is < 0.05: there is moderate evidence that the correlation is significant.
            p-value is < 0.1: there is weak evidence that the correlation is significant.
            p-value is >  0.1: there is no evidence that the correlation is significant.
'''

# p-value & pearson coeff with the .pearsonr() method. wheel.base vs price
# Since the p-value is < 0.001, the correlation between wheel-base and price is statistically significant,
# although the linear relationship isn't extremely strong (~0.585)
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# p-value & pearson coeff with the .pearsonr() method. horsepower vs price
# Since the p-value is < 0.001, the correlation between horsepower and price is statistically significant,
# and the linear relationship is quite strong (~0.809, close to 1).
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# p-value & pearson coeff with the .pearsonr() method. length vs price
# Since the p-value is < 0.001, the correlation between length and price is statistically significant,
# and the linear relationship is moderately strong (~0.691).
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# p-value & pearson coeff with the .pearsonr() method. width vs price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value )

# p-value & pearson coeff with the .pearsonr() method. curb-weight vs price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# p-value & pearson coeff with the .pearsonr() method. engine size vs price
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# p-value & pearson coeff with the .pearsonr() method. bore vs price
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value )

# p-value & pearson coeff with the .pearsonr() method. city-mpg vs price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# p-value & pearson coeff with the .pearsonr() method. highway-L/100km vs price
pearson_coef, p_value = stats.pearsonr(df['highway-L/100km'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

'''
    [] ANOVA : 
        The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences 
        between the means of two or more groups. ANOVA returns two parameters:

        F-test score: 
            ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate 
            from the assumption, and reports it as the F-test score. A larger score means there is a larger difference 
            between the means.

        P-value: 
            P-value tells how statistically significant our calculated score value is.

        If our price variable is strongly correlated with the variable we are analyzing, we expect ANOVA to return 
        a sizeable F-test score and a small p-value.
'''

# Since ANOVA analyzes the difference between different groups of the same variable, the groupby() function will come in handy.
# Because the ANOVA algorithm averages the data automatically, we do not need to take the average beforehand.
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)
print(grouped_test2)

#Finally what we need to do is use the f_oneway() function in the module stats to obtain the f-test scores and p-values.

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)


# ANOVA fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)


# ANOVA 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)


# ANOVA 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)




