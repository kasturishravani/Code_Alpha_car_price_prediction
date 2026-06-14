import pickle
import pandas as pd

# Load model
model = pickle.load(open("car_price_model.pkl", "rb"))

def predict_price(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return round(prediction[0], 2)
