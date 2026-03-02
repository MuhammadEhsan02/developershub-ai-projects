# ----------------------------------------
# House Price Prediction - Test Script
# ----------------------------------------
# This script loads the saved trained model, scaler, and label encoder,
# prepares new house input data, applies preprocessing, and predicts
# the estimated house price.

# Import required libraries
import joblib   # Used for loading the saved machine learning model and preprocessing objects
import pandas as pd   # Used for creating and handling structured input data in DataFrame format


# ----------------------------------------
# Load Saved Model and Preprocessing Objects
# ----------------------------------------

# Load the saved file which contains:
# - trained Linear Regression model
# - StandardScaler used during training
# - LabelEncoder used to encode categorical feature "Neighborhood"
data = joblib.load("outputs/house_price_model.pkl")

# Extract individual components from the loaded file
model = data["model"]              # Trained machine learning model used for prediction
scaler = data["scaler"]            # Scaler used to standardize input features
le = data["label_encoder"]         # LabelEncoder used to convert neighborhood names into numeric values


# ----------------------------------------
# Create New House Data for Prediction
# ----------------------------------------

# Create a pandas DataFrame containing the features of a new house
# This must match the same feature names and format used during training
new_house = pd.DataFrame({
    
    # Above ground living area in square feet
    "GrLivArea": [2000],
    
    # Number of bedrooms above ground level
    "BedroomAbvGr": [3],
    
    # Number of full bathrooms
    "FullBath": [2],
    
    # Neighborhood name (categorical feature)
    "Neighborhood": ["NAmes"]
})


# ----------------------------------------
# Encode Categorical Feature
# ----------------------------------------

# Convert neighborhood name into numeric label using the same LabelEncoder
# This ensures consistency with the training data encoding
new_house["Neighborhood"] = le.transform(new_house["Neighborhood"])


# ----------------------------------------
# Apply Feature Scaling
# ----------------------------------------

# Scale the input features using the same StandardScaler used during training
# This ensures the input is in the same scale as the training data
new_house_scaled = scaler.transform(new_house)


# ----------------------------------------
# Predict House Price
# ----------------------------------------

# Use the trained model to predict the house price
price = model.predict(new_house_scaled)


# ----------------------------------------
# Display Prediction Result
# ----------------------------------------

# Print the predicted house price
print("Predicted House Price:", price[0])