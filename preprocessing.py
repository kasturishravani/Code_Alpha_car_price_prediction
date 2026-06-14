import pandas as pd

def preprocess(df):

    df.drop_duplicates(inplace=True)

    df["Car_Age"] = 2025 - df["Year"]

    df = pd.get_dummies(
        df,
        columns=["Fuel_Type",
                 "Seller_Type",
                 "Transmission"],
        drop_first=True
    )

    return df
