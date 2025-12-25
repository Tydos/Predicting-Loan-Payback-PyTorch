import pandas as pd

def read_data():
    df = pd.read_csv("dataset/train.csv")
    df = df.sample(1000)
    return df