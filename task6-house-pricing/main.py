# 🏠 Task 6: House Price Prediction
# This project predicts house prices using basic features from the dataset. 
# It includes data cleaning, EDA, feature scaling, model training, evaluation, 
# visualization, and saving the trained model.

# ----------------------------
# Import Required Libraries
# ----------------------------

import os# For folder creation and file handling
import pandas as pd# For data loading and manipulation
import matplotlib.pyplot as plt# For basic plotting
import seaborn as sns# For advanced visualization
import numpy as np# For numerical operations

# Scikit-learn imports
from sklearn.model_selection import train_test_split # Split data into train/test sets
from sklearn.linear_model import LinearRegression # Linear regression model
from sklearn.metrics import mean_absolute_error, mean_squared_error# Model evaluation metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler # Encoding and scaling features


# ----------------------------
# Create Outputs Folder
# ----------------------------
# Ensure an 'outputs' folder exists to save plots and model files. This prevents errors when saving files later.
os.makedirs("outputs", exist_ok=True)


# ----------------------------
# Load Dataset
# ----------------------------
# Load the dataset CSV into a pandas DataFrame. Make sure the 'house_prices.csv' file is in the same directory as this script or provide the correct path.
df = pd.read_csv("house_prices.csv")
# Show the first 5 rows of the dataset to understand its structure and the types of data it contains.
print("\nFirst 5 rows:")# Display the first 5 rows of the dataset to get an overview of the data.
print(df.head())# Display dataset information to check data types and non-null counts.
# Display dataset information: column types, non-null counts, etc.
print("\nDataset Info:")# Check for missing values in each column to identify which columns need cleaning.
print(df.info())# Check for missing values in each column to identify which columns need cleaning.
# Check for missing values in the dataset to identify which columns have null values that need to be handled.
print("\nMissing Values:")# Check for missing values in each column to identify which columns need cleaning.
print(df.isnull().sum())# Display the count of missing values in each column to understand the extent of data cleaning required.


# ----------------------------
# Handle Missing Values
# ----------------------------

# Numeric columns
# Define numeric columns to fill missing values with the median. The median is used instead of the mean because it is less affected by outliers, which can skew the data.
numeric_cols = ["GrLivArea", "BedroomAbvGr", "FullBath"]
# Fill numeric columns' missing values with the median of each column. This is a common technique to handle missing values in numeric data, especially when the data may contain outliers.
# Median is less sensitive to outliers than mean, so it provides a better central tendency measure for skewed data.
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
# Fill missing categorical data (Neighborhood) with mode (most frequent value). This is a common technique for handling missing values in categorical data, as it replaces null values with the most common category, preserving the distribution of the data.
# Categorical column
df["Neighborhood"] = df["Neighborhood"].fillna(
    df["Neighborhood"].mode()[0]
)


# ----------------------------
# Encode Categorical Data
# ----------------------------
# Machine learning models require numeric inputs
# Convert 'Neighborhood' (categorical) to numeric using LabelEncoder. This transforms each unique category in the 'Neighborhood' column into a unique integer, allowing the model to process this feature.
le = LabelEncoder()# Initialize the LabelEncoder to convert categorical text data into numeric labels.

df["Neighborhood"] = le.fit_transform(df["Neighborhood"])# Fit the LabelEncoder to the 'Neighborhood' column and transform it into numeric labels. Each unique neighborhood will be assigned a unique integer value.


# ----------------------------
# Exploratory Data Analysis (EDA)
# ----------------------------

# Price Distribution
# Plot price distribution to understand target variable distribution. This helps to identify if the target variable (house prices) is skewed, which can affect model performance. A histogram with a KDE (Kernel Density Estimate) overlay provides insights into the distribution of house prices.
plt.figure(figsize=(8,5)) # Set the figure size for better visibility of the plot.

sns.histplot(df["SalePrice"], kde=True)# Create a histogram of the 'SalePrice' column with a KDE overlay to visualize the distribution of house prices. This helps to identify if the data is skewed or has outliers.

plt.title("House Price Distribution")# Set the title of the plot to "House Price Distribution" to indicate what the plot represents.

plt.xlabel("Sale Price")# Set the x-axis label to "Sale Price" to indicate that the x-axis represents the house prices.
plt.ylabel("Count")# Set the y-axis label to "Count" to indicate that the y-axis represents the frequency of house prices in each bin.

plt.savefig("outputs/price_distribution.png")# Save the price distribution plot as a PNG file in the 'outputs' folder for later reference.

plt.show()# Display the price distribution plot to visually analyze the distribution of house prices in the dataset.


# ----------------------------
# Correlation Heatmap
# ----------------------------
# Correlation heatmap to see relationships between numeric features. This helps to identify which features are strongly correlated with the target variable (SalePrice) and with each other. A heatmap provides a visual representation of these correlations, making it easier to spot patterns and relationships in the data.
numeric_df = df.select_dtypes(include=[np.number])# Select only numeric columns from the DataFrame to create a correlation heatmap, as correlation is only meaningful for numeric data. This ensures that the heatmap will only include relevant features for analysis.

plt.figure(figsize=(12,10))# Set the figure size for better visibility of the heatmap, especially when there are many numeric features to display.

sns.heatmap(# Create a heatmap to visualize the correlation between numeric features in the dataset. The 'annot=True' argument adds the correlation values to each cell, and 'cmap="coolwarm"' sets the color scheme for better visual distinction between positive and negative correlations.
    numeric_df.corr(), # Compute the correlation matrix of the numeric features to understand how they relate to each other and to the target variable.
    cmap="coolwarm"# Use the 'coolwarm' color map to visually differentiate between positive and negative correlations, where warm colors indicate positive correlation and cool colors indicate negative correlation.
)

plt.title("Correlation Heatmap (Numeric Features Only)")# Set the title of the heatmap to "Correlation Heatmap (Numeric Features Only)" to indicate that the heatmap shows correlations between numeric features in the dataset.

plt.savefig("outputs/correlation_heatmap.png")# Save the correlation heatmap as a PNG file in the 'outputs' folder for later reference and analysis.

plt.show()# Display the correlation heatmap to visually analyze the relationships between numeric features and identify which features are strongly correlated with the target variable (SalePrice) and with each other.


# ----------------------------
# Feature Selection
# ----------------------------
# Select features for model training. Based on the correlation heatmap and domain knowledge, we select 'GrLivArea', 'BedroomAbvGr', 'FullBath', and 'Neighborhood' as input features (X) to predict the target variable (y), which is 'SalePrice'. This step is crucial for improving model performance by using relevant features.
X = df[[
    "GrLivArea",# Above ground living area in square feet, which is often strongly correlated with house price.
    "BedroomAbvGr",# Number of bedrooms above ground, which can influence the price of a house.
    "FullBath",# Number of full bathrooms, which is an important feature for house valuation.
    "Neighborhood"# Encoded neighborhood feature, which captures location-based differences in house prices.
]]
# Target variable (house price)
y = df["SalePrice"]# The target variable we want to predict, which is the sale price of the house.


print("\nSelected Features:")# Print the first few rows of the selected features to verify that we have the correct columns for model training.
print(X.head())# Display the first 5 rows of the selected features to confirm that we have the correct data for model training. This helps to ensure that the feature selection step was successful and that we are using the intended features for prediction.


# ----------------------------
# Feature Scaling
# ----------------------------
# Scale features to improve model performance. Feature scaling is important for many machine learning algorithms, especially those that rely on the distance between data points (like linear regression). StandardScaler standardizes features by removing the mean and scaling to unit variance, which can help improve model convergence and performance.
scaler = StandardScaler()# Initialize the StandardScaler to standardize the features by removing the mean and scaling to unit variance. This helps to ensure that all features are on the same scale, which can improve the performance of the machine learning model. # StandardScaler: mean=0, std=1

X_scaled = scaler.fit_transform(X)# Fit the StandardScaler to the selected features (X) and transform them into a scaled version (X_scaled). This step standardizes the features, making them more suitable for model training and improving the performance of algorithms that are sensitive to feature scales.


# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(# Split the scaled features (X_scaled) and target variable (y) into training and testing sets. The test size is set to 20% of the data, and a random state is provided for reproducibility. This step is crucial for evaluating the model's performance on unseen data.

    X_scaled,# Scaled input features for training and testing.
    y,# Target variable (house prices) for training and testing.
    test_size=0.2,# 20% of the data will be used for testing, and 80% will be used for training the model. This allows us to evaluate the model's performance on unseen data.
    random_state=42# Setting a random state ensures that the train-test split is reproducible, meaning that the same split will occur each time the code is run, which is important for consistent model evaluation.

)

print("\nTraining samples:", len(X_train))# Print the number of training samples to confirm that the train-test split was successful and to understand the size of the training dataset.
print("Testing samples:", len(X_test))# Print the number of testing samples to confirm that the train-test split was successful and to understand the size of the testing dataset. This helps to ensure that we have enough data for both training and evaluating the model.


# ----------------------------
# Train Model
# ----------------------------

model = LinearRegression()# Initialize the Linear Regression model, which will be used to learn the relationship between the input features and the target variable (house prices). Linear regression is a simple and interpretable model that assumes a linear relationship between the features and the target.

model.fit(X_train, y_train)# Fit the Linear Regression model to the training data (X_train and y_train). This step allows the model to learn the coefficients that best fit the relationship between the input features and the target variable (house prices) based on the training data.

print("\nModel trained successfully.")# Print a message to indicate that the model has been trained successfully, confirming that the fitting process is complete and the model is ready for making predictions.


# ----------------------------
# Make Predictions
# ----------------------------
# Predict house prices on test data
y_pred = model.predict(X_test) # Use the trained Linear Regression model to make predictions on the test set (X_test). This step generates predicted house prices (y_pred) based on the input features in the test set, allowing us to evaluate the model's performance by comparing these predictions to the actual house prices (y_test).

print("\nSample Predictions:")# Print the first 5 predicted house prices to get an idea of the model's predictions and to verify that the prediction step is working correctly. This can help to identify if the predictions are reasonable based on the scale of house prices in the dataset.
print(y_pred[:5])# Display the first 5 predicted house prices to verify that the model is generating predictions and to get a sense of the predicted values compared to actual house prices in the dataset. This can help to identify if the predictions are in a reasonable range based on the scale of house prices in the dataset.


# ----------------------------
# Evaluate Model
# ----------------------------
# Mean Absolute Error (MAE) - average absolute prediction error
mae = mean_absolute_error(y_test, y_pred)# Calculate the Mean Absolute Error (MAE) between the actual house prices (y_test) and the predicted house prices (y_pred). MAE is a common evaluation metric for regression models that measures the average absolute difference between predicted and actual values, providing insight into the average prediction error.
# Root Mean Squared Error (RMSE) - penalizes large errors more than MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))# Calculate the Root Mean Squared Error (RMSE) between the actual house prices (y_test) and the predicted house prices (y_pred). RMSE is another common evaluation metric for regression models that measures the square root of the average squared differences between predicted and actual values, giving more weight to larger errors compared to MAE.


print("\nModel Performance:")# Print a header to indicate that the following output will show the performance metrics of the model, providing a clear separation from previous outputs and highlighting the evaluation results.

print("Mean Absolute Error (MAE):", round(mae, 2))# Print the Mean Absolute Error (MAE) rounded to 2 decimal places to provide a clear and concise measure of the average prediction error of the model. This helps to understand how well the model is performing in terms of average absolute error.

print("Root Mean Squared Error (RMSE):", round(rmse, 2))# Print the Root Mean Squared Error (RMSE) rounded to 2 decimal places to provide a clear and concise measure of the model's performance, especially in terms of penalizing larger errors. This helps to understand how well the model is performing in terms of average squared error, giving more weight to larger errors compared to MAE.


# ----------------------------
# Actual vs Predicted Plot
# ----------------------------
# Scatter plot of actual vs predicted house prices to visualize model performance. This plot helps to visually assess how well the predicted values match the actual values. Ideally, the points should lie close to a diagonal line (y = x) if the predictions are accurate.
plt.figure(figsize=(8,6))# Set the figure size for better visibility of the scatter plot, allowing for a clearer visualization of the relationship between actual and predicted house prices.

plt.scatter(y_test, y_pred)# Create a scatter plot with actual house prices (y_test) on the x-axis and predicted house prices (y_pred) on the y-axis. This visualization helps to assess the accuracy of the model's predictions by showing how closely the predicted values match the actual values.

plt.xlabel("Actual Price")# Set the x-axis label to "Actual Price" to indicate that the x-axis represents the actual house prices from the test set.

plt.ylabel("Predicted Price")# Set the y-axis label to "Predicted Price" to indicate that the y-axis represents the predicted house prices generated by the model.

plt.title("Actual vs Predicted Prices")# Set the title of the plot to "Actual vs Predicted Prices" to indicate that the plot shows the relationship between actual and predicted house prices, allowing for a visual assessment of model performance.

plt.savefig("outputs/actual_vs_predicted.png")# Save the actual vs predicted scatter plot as a PNG file in the 'outputs' folder for later reference and analysis.

plt.show()# Display the actual vs predicted scatter plot to visually analyze the model's performance and assess how well the predicted house prices match the actual house prices in the test set. This can help to identify patterns, trends, and potential areas for improvement in the model.


# ----------------------------
# Feature Importance
# ----------------------------
# Coefficients from Linear Regression indicate feature importance. In linear regression, the coefficients represent the change in the target variable (house price) for a one-unit change in the feature, holding all other features constant. By examining the coefficients, we can understand which features have a stronger influence on the predicted house prices.
importance = pd.Series(# Create a pandas Series to represent the feature importance based on the coefficients of the trained Linear Regression model. The index of the Series corresponds to the feature names, and the values correspond to the coefficient values, which indicate the importance of each feature in predicting house prices.

    model.coef_,# The coefficients from the trained Linear Regression model, which indicate the importance of each feature in predicting house prices. A higher absolute value of a coefficient indicates a stronger influence on the target variable.
    index=[# The names of the features corresponding to the coefficients, which will be used as the index for the Series to make it easier to interpret the feature importance.
        "GrLivArea",# Above ground living area in square feet, which is often strongly correlated with house price.
        "BedroomAbvGr",# Number of bedrooms above ground, which can influence the price of a house.
        "FullBath",# Number of full bathrooms, which is an important feature for house valuation.
        "Neighborhood"# Encoded neighborhood feature, which captures location-based differences in house prices.
    ]

)
# Plot feature importance based on coefficients. A horizontal bar plot is used to visualize the importance of each feature, making it easier to compare the influence of different features on the predicted house prices.
importance.sort_values().plot(# Sort the feature importance values and create a horizontal bar plot to visualize the importance of each feature in predicting house prices. Sorting the values helps to clearly show which features are more important than others.

    kind="barh",# Create a horizontal bar plot to visualize feature importance, which allows for easier comparison of feature importance values, especially when there are multiple features to display.
    figsize=(8,5)# Set the figure size for better visibility of the feature importance plot, allowing for a clearer visualization of the importance of each feature in predicting house prices.

)

plt.title("Feature Importance")# Set the title of the plot to "Feature Importance" to indicate that the plot shows the importance of each feature in predicting house prices based on the coefficients from the Linear Regression model.

plt.xlabel("Coefficient Value")# Set the x-axis label to "Coefficient Value" to indicate that the x-axis represents the coefficient values from the Linear Regression model, which indicate the importance of each feature in predicting house prices.

plt.savefig("outputs/feature_importance.png")# Save the feature importance plot as a PNG file in the 'outputs' folder for later reference and analysis.

plt.show()# Display the feature importance plot to visually analyze which features have a stronger influence on the predicted house prices based on the coefficients from the Linear Regression model. This can help to identify which features are most important for predicting house prices and may provide insights for feature selection or further analysis.


# ----------------------------
# Model Accuracy (R² Score)
# ----------------------------
# R² score indicates percentage of variance explained by the model. The R² score, also known as the coefficient of determination, measures how well the model's predictions match the actual data. It represents the proportion of the variance in the target variable (house prices) that is explained by the features used in the model. An R² score of 1 indicates perfect predictions, while an R² score of 0 indicates that the model does not explain any of the variance in the target variable.
score = model.score(X_test, y_test)# Calculate the R² score of the trained Linear Regression model on the test set (X_test and y_test). This score indicates how well the model's predictions match the actual house prices in the test set, providing insight into the percentage of variance in house prices that is explained by the features used in the model.

print("\nModel Accuracy (R² Score):", round(score * 100, 2), "%")# Print the R² score as a percentage, rounded to 2 decimal places, to provide a clear and concise measure of the model's accuracy in explaining the variance in house prices. This helps to understand how well the model is performing in terms of its predictive power, with higher percentages indicating better performance.

# ----------------------------
# Save Model and Scaler
# ----------------------------
# Library to save/load Python objects
import joblib# Import the joblib library, which is used for saving and loading Python objects, including machine learning models. This allows us to save the trained model to a file and load it later without having to retrain it.

# Save both the trained model and the scaler together
joblib.dump({# Create a dictionary to store the trained model, scaler, and label encoder together. This allows us to save all necessary components for making predictions in the future, ensuring that we can easily load the model and its associated preprocessing steps when needed.
    "model": model,# The trained Linear Regression model that we want to save for future use. This model can be loaded later to make predictions on new data without having to retrain it.
    "scaler": scaler,# The StandardScaler used for feature scaling during model training. Saving the scaler allows us to apply the same scaling to new data when making predictions, ensuring consistency in the input features.
    "label_encoder": le# The LabelEncoder used for encoding the 'Neighborhood' categorical feature. Saving the label encoder allows us to apply the same encoding to new data when making predictions, ensuring that the categorical feature is processed in the same way as during training.
}, "outputs/house_price_model.pkl")# Save the dictionary containing the model, scaler, and label encoder to a file named 'house_price_model.pkl' in the 'outputs' folder. This file can be loaded later to access the trained model and its associated preprocessing components for making predictions on new data.

print("\nModel, scaler, and label encoder saved successfully!")# Print a message to indicate that the model, scaler, and label encoder have been saved successfully, confirming that the saving process is complete and that the necessary components for making predictions in the future are now stored in the specified file. This provides assurance that the trained model and its associated preprocessing steps can be easily accessed later when needed.

# ----------------------------
# Final Message
# ----------------------------

print("\n✅ TASK COMPLETE!")# Print a final message to indicate that the entire task of house price prediction has been completed successfully, providing a clear conclusion to the script and signaling that all steps from data loading to model evaluation and saving have been executed without errors.
print("House Price Prediction Model created successfully.")# Print a message to indicate that the house price prediction model has been created successfully, summarizing the outcome of the task and confirming that the model is ready for use.
print("Check 'outputs' folder for plots and model file.")# Print a message to inform the user that the generated plots and the saved model file can be found in the 'outputs' folder, providing guidance on where to look for the results of the task. This helps to ensure that the user knows where to find the outputs generated by the script, including visualizations and the trained model.
