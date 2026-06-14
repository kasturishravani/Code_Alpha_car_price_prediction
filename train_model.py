import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("dataset/car_data.csv")

# Remove car name column
df.drop("Car_Name", axis=1, inplace=True)

# Convert categorical columns into numbers
df = pd.get_dummies(
    df,
    columns=[
        "Fuel_Type",
        "Seller_Type",
        "Transmission"
    ],
    drop_first=True
)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model Saved Successfully")
