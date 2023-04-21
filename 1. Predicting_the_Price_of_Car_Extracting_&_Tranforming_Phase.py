import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



''' 
(1) In every data importation you need to follow these easy steps: 
    1. Extracting:
        from different sources.
    2. Transforming:
        Normalizing, type converting, cleaning, ...
    3. Loading:
        Into models to produce predictable results from a given target.  
'''

######################################### Extracting Phase ##############################################################

''' 
read the downloaded file using the file path and assign it to variable "df"
this when you use the pd.read_csv() you are created a DataFrame and as such you can call the df.head() to display the
first 5 rows of the data.
'''

path = 'C:/Users/abeal/Desktop/work/Predicting Models/Predicting car prices/imports-85.data'
df = pd.read_csv(path, header=None)
df.head()

# We can see from the output of df.head() that the df doesn't have any headers, and as such we must give them one.
# Check afterwards
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns = headers
df.head(20)

######################################### Transforming Phase ############################################################

'''
(2) Replacing the ? for NotaNumber (NaN):
    Now what we need to do is replace the "?" for "NaN" in the df by using a df.replace() function so that
    We can drop the rows that have missing values with the function df1.dropna() along the column "price" as follows:
'''
df1 = df.replace('?', np.NaN)
df1.head()
df = df1.dropna(subset=["price"], axis=0)
df.head(20)

'''
(3) Dropping the NaN for our target:
    After clearing out all the rows that have NaN (Not a Number, which are missing values) for our target output --> price
    we should check the types of information we have imported and see if it makes coherence sense.
'''
print(df.dtypes)

# Now what will do is a statistical analysis of the dataframe excluding the NaN>
df.describe()
# However, what if we would also like to check all the columns including those that are of type object?
print(df.describe(include='all'))

# We are now going to evaluate the missing data in the rest of the columns: False == No missing data; True == missing data
missing_data = df.isnull()
missing_data.head(5)

'''
seeing that there are missing data, we need to count how many per column:
     .columns returns a list with the names of the columns
     .value_counts() counts the unique values inside a columns
'''
for column in missing_data.columns:
    print(column)
    print(missing_data[column].value_counts())
    print("")

'''
Based on the summary above, each column has 201 rows of data and seven of the columns containing missing data:

    "normalized-losses": 41 missing data
    "num-of-doors": 2 missing data
    "bore": 4 missing data
    "stroke" : 4 missing data
    "horsepower": 2 missing data
    "peak-rpm": 2 missing data
    "price": 4 missing data'''

'''
(4) Deal with missing data:
    How to deal with missing data?

    Drop data
        a. Drop the whole row
        b. Drop the whole column
    Replace data
        a. Replace it by mean
        b. Replace it by frequency
        c. Replace it based on other functions 

    Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns 
    are empty enough to drop entirely. We have some freedom in choosing which method to replace data; however, 
    some methods may seem more reasonable than others. We will apply each method to many different columns:

    Replace by mean:

        "normalized-losses": 41 missing data, replace them with mean
        "stroke": 4 missing data, replace them with mean
        "bore": 4 missing data, replace them with mean
        "horsepower": 2 missing data, replace them with mean
        "peak-rpm": 2 missing data, replace them with mean

    Replace by frequency:

        "num-of-doors": 2 missing data, replace them with "four".
            Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur

    Drop the whole row:

        "price": 4 missing data, simply delete the whole row
            Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; 
            therefore any row now without price data is not useful to us
'''

# Replace by mean ---> Calculate the mean value for the "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

# Replace "NaN" with mean value in "normalized-losses" column
df.loc[df["normalized-losses"].isnull(), "normalized-losses"] = avg_norm_loss

# Replace by mean ---> Calculate the mean value for the "bore" column
avg_bore = df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

# Replace "NaN" with the mean value in the "bore" column
df.loc[df["bore"].isnull(), "bore"] = avg_bore

# Replace by mean ---> Calculate the mean value for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)

# Replace NaN by mean value in "stroke" column
df.loc[df["stroke"].isnull(), "stroke"] = avg_stroke

# Replace by mean ---> Calculate the mean value for "horsepower" column
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", avg_horsepower)

# Replace NaN by mean value in "horsepower" column
df.loc[df["horsepower"].isnull(), "horsepower"] = avg_horsepower

# Replace by mean ---> Calculate the mean value for "peak-rpm" column
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of peak-rpm:", avg_peak_rpm)

# Replace NaN by mean value in "peak-rpm" column
df.loc[df["peak-rpm"].isnull(), "peak-rpm"] = avg_peak_rpm

# Replace by frequency ---> See which strings are the most common for "num-of-doors" column:
df['num-of-doors'].value_counts()

# We can see that four doors are the most common type. We can also use the ".idxmax()"
# method to calculate the most common type automatically:
most_common_num_of_doors = df['num-of-doors'].value_counts().idxmax()
print(most_common_num_of_doors)

# The replacement procedure is very similar to what we have seen previously
# replace the missing 'num-of-doors' values by the most frequent
df.loc[df["num-of-doors"].isnull(), "num-of-doors"] = most_common_num_of_doors

# Now we have a DataSet with no missing values. Hurray!!!

'''
(5) Correct data format:
    We are almost there!
        The last step in data cleaning is checking and making sure that all data is in the correct format 
        (int, float, text or other).

    In Pandas, we use:
        .dtype() to check the data type
    .   astype() to change the data type
'''
print(df.dtypes)

'''As we can see above, some columns are not of the correct data type. 
Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. 
For example, 'bore' and 'stroke' variables are numerical values that describe the engines, so we should 
expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'. 
We have to convert data types into a proper format for each column using the "astype()" method.
'''

# Convert data types
numeric_cols = ['bore', 'stroke', 'price', 'peak-rpm']
df.loc[:, numeric_cols] = df[numeric_cols].astype(float)
df.loc[:, 'normalized-losses'] = df['normalized-losses'].astype(int)

# Print data types
print(df.dtypes)

# we have now the format all correct, no NaN (or missing values) and the right Type per column

'''
'''

'''
(6) Data Standardization
    What is standardization?
        Standardization is the process of transforming data into a common format, allowing the researcher to make the 
        meaningful comparison.

    Common approaches:
        Data is usually collected from different agencies in different formats. 
        (Data standardization is also a term for a particular type of data normalization where we subtract the mean and 
        divide by the standard deviation.)

    Example:
        Transform mpg to L/100km:
'''
# transform mpg to L/100km by mathematical operation (235 divided by mpg), we use the .loc method to edit the original df
df.loc[:,"highway-mpg"] = 235 / df["highway-mpg"]

# In the first line, we use .assign() to create a new column 'highway-L/100km' with values from the 'highway-mpg' column
# This modifies the original dataframe directly and avoids creating a copy of the data.
# In the second line, we drop the 'highway-mpg' column using df.drop(). Again, we modify the original dataframe directly
# "highway-mpg" to "highway-L/100km"
df = df.assign(**{'highway-L/100km': df['highway-mpg']}).drop('highway-mpg', axis=1)

# check your transformed data
print(df.head())

'''
(7) Data Normalization
    Why normalization?
        Normalization is the process of transforming values of several variables into a similar range. Typical 
        normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance 
        is 1, or scaling the variable so the variable values range from 0 to 1.

    Example
        To demonstrate normalization, let's say we want to scale the columns "length", "width" and "height".

    Target: 
        would like to normalize those variables so their value ranges from 0 to 1

    Approach: 
        replace original value by (original value)/(maximum value)
'''
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / df['height'].max()
df[["length", "width", "height"]].head()

'''
(8) Binning
    Why binning?
        Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' 
        for grouped analysis.

    Example:
        In our dataset, "horsepower" is a real valued variable ranging from 48 to 288 and it has 59 unique values. 
        What if we only care about the price difference between cars with high horsepower, medium horsepower, and little 
        horsepower (3 types)? Can we rearrange them into three ‘bins' to simplify analysis?

    We will use the pandas method 'cut' to segment the 'horsepower' column into 3 bins.
'''
# Example of binning: We will start by converting the column into the correct format.
df["horsepower"] = df["horsepower"].astype(int, copy=True)

# We plot the histogram to see how the distribution looks like:
plt.hist(df["horsepower"])

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

'''We would like 3 bins of equal size bandwidth so we use numpy's linspace(start_value, end_value, numbers_generated function.
Since we want to include the minimum value of horsepower, we want to set start_value = min(df["horsepower"]).
Since we want to include the maximum value of horsepower, we want to set end_value = max(df["horsepower"]).
Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated = 4.'''

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

# we set group names:
group_names = ['Low', 'Medium', 'High']

# We apply the function "cut" to determine what each value of df['horsepower'] belongs to.
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
print(df[['horsepower', 'horsepower-binned']].head(20))

# Let's see the number of vehicles in each bin:
df["horsepower-binned"].value_counts()

# Let's plot the distribution of each bin:
plt.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

''''
Bins Visualization
Normally, a histogram is used to visualize the distribution of bins we created above.
'''
# draw historgram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins=3)

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

'''
(9) Indicator Variable (or Dummy Variable)
    What is an indicator variable?
        An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 
        'dummies' because the numbers themselves don't have inherent meaning.

    Why we use indicator variables?
        We use indicator variables so we can use categorical variables for regression analysis in the later modules.

    Example
        We see the column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, 
        only numbers. To use this attribute in regression analysis, we convert "fuel-type" to indicator variables.

        We will use pandas' method 'get_dummies' to assign numerical values to different categories of fuel type.
'''
# we are going to transform into dummies 2 columns: 1) fuel.type 2) aspiration
# let's see the names of the columns:
print(df.columns)

# fuel-type: get the indicator variable and assign it to the dataframe 'dummy_variable_1'
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

# fuel-type:  change the columnn name for clarity:
dummy_variable_1.rename(columns={'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel'}, inplace=True)
print(dummy_variable_1.head())

# fuel-type:  merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# fuel-type:  drop original column "fuel-type" from "df"
df.drop("fuel-type", axis=1, inplace=True)
print(df.head())

# aspiration: Get the indicator variables for "aspiration"and assign it to data frame "dummy_variable_2":
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.head()

# aspiration: change the name for clarity
dummy_variable_2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
dummy_variable_2.head()

# aspiration: merge data frame "df" and "dummy_variable_2"
df = pd.concat([df, dummy_variable_1], axis=1)

# aspiration: drop original column "fuel-type" from "df"
df.drop("aspiration", axis=1, inplace=True)

# saving it now to a csv file:
df.to_csv('clean_df.csv')


