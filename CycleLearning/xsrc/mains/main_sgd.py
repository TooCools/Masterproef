import time

from xsrc.analyze import visualize_data
from xsrc.excel_to_data import get_data
from xsrc.learner import learn_PA, learn_SGD
from xsrc.params import *
from xsrc.preprocessing import preprocess

df_main = get_data("..\\..\\data\\data.xlsx", [df_torque, df_crank_angle_rad, df_fcc])
df_val = get_data("..\\..\\data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_fcc])

print("Preprocessing data")
train_x, train_y = preprocess(df_main, 50, normalize=True)  # , classification=True)
val_x, val_y = preprocess(df_val, 50, normalize=True, shuffle=False)  # , classification=True)

print(val_x)
print(val_y)

model = learn_SGD("sgd", train_x, train_y, 10)  # ,classification=True)
predictions = model.predict(val_x)


visualize_data([predictions, val_y], ["predictions", "actual"])
print(model.score(val_x, val_y))
