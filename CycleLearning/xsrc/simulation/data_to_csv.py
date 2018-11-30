import pandas as pd


def save(data, filename="data"):
    print("Saving Data")
    df = pd.DataFrame(data,
                      columns=data.keys())
    writer = pd.ExcelWriter("..//..//data//"+filename + ".xlsx")
    df.to_excel(writer, "Bicycle data", index=False)
    writer.save()
