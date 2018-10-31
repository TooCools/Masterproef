import pandas as pd

data_path = "..\\Fietssimulatie\\data.xlsx"
sheet_name = "Bicycle Data"


def get_data(columns):
    df = pd.read_excel(data_path)
    cols_total = set(df.columns)
    diff = cols_total - set(columns)
    df.drop(diff, axis=1, inplace=True)
    return df
