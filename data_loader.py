import pandas as pd

def load_data():
    df = pd.read_csv("dataset/car_data.csv")
    return df
