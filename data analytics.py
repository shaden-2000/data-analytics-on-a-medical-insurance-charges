#the picture
'''
Setup
For this lab, we will be using the following libraries:
skillsnetwork to download the data
pandas for managing the data.
numpy for mathematical operations.
sklearn for machine learning and machine-learning-pipeline related functions.
seaborn for visualizing the data.
matplotlib for additional plotting tools.'''


import piplite
await piplite.install('seaborn')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split



filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath, header=None)

#Task 1 : Import the dataset
'''Import the dataset into a pandas dataframe. Note that there are currently no headers in the CSV file.
Print the first 10 rows of the dataframe to confirm successful loading.'''

df = pd.read_csv(filepath, header=None)
print(df.head(10))

#Add the headers to the dataframe, as mentioned in the project scenario.

headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers

#Now, replace the '?' entries with 'NaN' values.
df.replace('?', np.nan, inplace = True)


#Task 2 : Data Wrangling
#Use `dataframe.info()` to identify the columns that have some 'Null' (or NaN) information.
print(df.info())

'''
Handle missing data:
For continuous attributes (e.g., age), replace missing values with the mean.
For categorical attributes (e.g., smoker), replace missing values with the most frequent value.
Update the data types of the respective columns.
Verify the update using df.info().
'''

is_smoker = df['smoker'].value_counts().idxmax()
df["smoker"].replace(np.nan, is_smoker, inplace=True)

# age is a continuous variable, replace with mean age
mean_age = df['age'].astype('float').mean(axis=0)
df["age"].replace(np.nan, mean_age, inplace=True)

# Update data types
df[["age","smoker"]] = df[["age","smoker"]].astype("int")

print(df.info())

#Update the charges column such that all values are rounded to nearest 2 decimal places
df[["charges"]] = np.round(df[["charges"]],2)


#Exploratory Data Analysis (EDA) + picture
Implement the regression plot for charges with respect to bmi.
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color": "red"})
#Implement the box plot for charges with respect to smoker. + picture
sns.boxplot(x="smoker", y="charges", data=df)