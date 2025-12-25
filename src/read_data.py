import pandas as pd

def read_data(data_length: int) -> pd.DataFrame:
    df = pd.read_csv("dataset/train.csv")
    df = df.sample(n=data_length)
    return df
