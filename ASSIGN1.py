

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


df = pd.read_csv('GlobalSuperstore.csv')

df.head()
df.info()
df.describe()

# Checking for missing values
df.isnull().sum()

df.drop_duplicates(inplace=True)

# Converting date columns to datetime type
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Descriptive statistics for numerical columns
df.describe()

# Summary statistics for categorical columns
df.describe(include='object')

# creating a histogram
plt.hist(df['Sales'])
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# creating a box plot
sns.boxplot(x='Category', y='Sales', data=df)
plt.title('Sales by Category')
plt.show()

# creating a count plot
sns.countplot(x='Category', data=df)
plt.title('Count of Orders by Category')
plt.show()


# creating a scatter plot
plt.scatter(df['Sales'], df['Profit'])
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.title('Sales vs. Profit')
plt.show()

# creating a correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# creating pair plots
sns.pairplot(df)
plt.title('Pair Plot')
plt.show()

# using pivot table
pivot_table = df.pivot_table(index='Category', columns='Sub-Category', values='Sales', aggfunc=np.mean)
print(pivot_table)

# using groupby
grouped_data = df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'mean', 'Profit': 'mean'})
print(grouped_data)


#DATA VISUALIZATION

sns.barplot(x='Category', y='Sales', data=df)
plt.title('Average Sales by Category')
plt.show()

sns.boxplot(x='Region', y='Profit', data=df)
plt.title('Profit by Region')
plt.show()

#FEATURE ENGINEERING

df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100
df['Sales per Customer'] = df['Sales'] / df['Customer ID'].nunique()


