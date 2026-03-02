# 🌸 Task 1: Exploring and Visualizing Iris Dataset

## 📌 Objective
# The goal of this task is to explore, analyze, and visualize the Iris dataset to understand feature distributions, relationships, and patterns.
# A simple machine learning model is also applied to evaluate data suitability.


# Import pandas library for data handling and analysis
# pandas is used to read CSV, Excel files and work with tables (DataFrames)
import pandas as pd


# Import matplotlib library for creating basic graphs and plots
# pyplot module helps create line charts, bar charts, etc.
import matplotlib.pyplot as plt


# Import seaborn library for advanced and beautiful data visualization
# seaborn is built on matplotlib and makes plots more attractive and easier
import seaborn as sns

# Load the built-in Iris dataset from seaborn library
# This dataset will be stored in variable df as a pandas DataFrame
df = sns.load_dataset("iris")

# -----------------------------
# Data Inspection
# -----------------------------

# Print the shape of the dataset
# df.shape returns (number_of_rows, number_of_columns)
print("Shape:", df.shape)

# Print the column names of the dataset
# df.columns returns all the column headers
print("Columns:", df.columns)

# Print the first 5 rows of the dataset
# df.head() shows top 5 records by default
print("\nFirst 5 Rows:\n")
print(df.head())

# Print detailed information about the dataset
# df.info() shows column names, non-null counts, data types
print("\nDataset Info:\n")
df.info()

# Print summary statistics for numeric columns
# df.describe() gives count, mean, std, min, 25%, 50%, 75%, max
print("\nSummary Statistics:\n")
print(df.describe())


# -----------------------------
# Scatter Plot
# -----------------------------

# Set figure size (width=6 inches, height=4 inches)
plt.figure(figsize=(6, 4))

# Create scatter plot using seaborn
# x-axis: sepal_length
# y-axis: sepal_width
# hue: species (different colors for each flower type)
# data: df (our Iris dataset)
sns.scatterplot(
    x="sepal_length",
    y="sepal_width",
    hue="species",
    data=df
)

# Add a title to the plot
plt.title("Sepal Length vs Sepal Width")

# Save the plot as a PNG file inside 'outputs' folder
plt.savefig("outputs/scatter_plot.png")

# Display the plot
plt.show()

# -----------------------------
# Histograms
# -----------------------------

# Create histograms for all numeric columns in df
# figsize=(8, 6) sets the size of the overall figure
df.hist(figsize=(8, 6))

# Add a main title for all subplots
plt.suptitle("Feature Distributions")

# Save the histogram figure in 'outputs' folder
plt.savefig("outputs/histograms.png")

# Display the histograms
plt.show()



# -----------------------------
# Box Plot
# -----------------------------

# Set figure size
plt.figure(figsize=(8, 4))

# Create a box plot for numeric columns (drop 'species' column)
# Box plot helps detect outliers and understand data spread
sns.boxplot(data=df.drop(columns="species"))

# Add a title to the plot
plt.title("Box Plot for Outlier Detection")

# Save the plot inside 'outputs' folder
plt.savefig("outputs/box_plot.png")

# Display the plot
plt.show()


