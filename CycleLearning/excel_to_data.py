import pandas as pd


def get_data(path, columns):
    df = pd.read_excel(path)
    cols_total = set(df.columns)
    diff = cols_total - set(columns)
    df.drop(diff, axis=1, inplace=True)
    return df
