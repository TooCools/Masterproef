import pandas as pd


def save(data):
    df = pd.DataFrame(data,
                      columns=data.keys())
    writer = pd.ExcelWriter('data.xlsx')
    df.to_excel(writer, "Bicycle data", index=False)
    writer.save()
