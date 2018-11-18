import time

from xsrc.analyze import visualize_data
from xsrc.excel_to_data import get_data
from xsrc.learner import learn_PA
from xsrc.params import *
from xsrc.preprocessing import preprocess
import numpy as np

df_main = get_data("..\\..\\data\\data.xlsx", [df_torque, df_crank_angle_rad, df_fcc])
df_val = get_data("..\\..\\data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_fcc])

print("Preprocessing data")
train_x, train_y = preprocess(df_main, 50, normalize=False)  # , classification=True)
val_x, val_y = preprocess(df_val, 50, normalize=False, shuffle=False)  # , classification=True)

model = learn_PA("patest", train_x, train_y, 10)  # ,classification=True)
predictions = model.predict(val_x)

visualize_data([predictions, val_y], ["predictions", "actual"])
print(model.score(val_x, val_y))

changed_x = []
changed_y = []

changed_x.extend(val_x[9000:9050])
changed_y.extend(val_y[9000:9050])

for (i, item) in enumerate(changed_y):
    changed_y[i] = item + 20

test = train_x.tolist()
test.extend(changed_x)
train_y.extend(changed_y)

start=time.time()
model.fit(test, train_y)
# model.fit(changed_x, changed_y)
print(time.time()-start)


predictions2 = model.predict(val_x)
visualize_data([predictions2, val_y], ["predictions", "actual"])
