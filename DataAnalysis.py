from ucimlrepo import fetch_ucirepo
import pandas as pd

# Load the Iris dataset
iris = fetch_ucirepo(id=53)  # Iris dataset has ID 53
df = iris.data.original

# Preview the data
print(df.head())

print(df.info())
print("\nMissing values:\n", df.isnull().sum())


df = df.dropna() 


# Describe numerical columns
print(df.describe())

# Group by species and get average measurements
grouped = df.groupby('class').mean(numeric_only=True)
print(grouped)

# Example insight
print("\nObservation: Setosa has shorter sepals and petals than Virginica.")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Line chart (simulate time series with index)
df['index'] = range(len(df))
plt.figure(figsize=(8, 4))
plt.plot(df['index'], df['sepal length'], label='Sepal Length')
plt.title('Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.legend()
plt.show()

# 2. Bar chart: Avg petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='class', y='petal length', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.show()

# 3. Histogram
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width'], bins=10, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length', y='petal length', hue='class', data=df)
plt.title('Sepal vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.show()

try:
    df = pd.read_csv('non_existent.csv')
except FileNotFoundError:
    print("⚠️ CSV file not found. Please check the filename or path.")




