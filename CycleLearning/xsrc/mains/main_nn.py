from xsrc.excel_to_data import get_data
from xsrc.stuff.learner import learn_lstm
from xsrc.stuff.params import *
from xsrc.stuff.preprocessing import preprocess
from xsrc.analyze import visualize_data

import time

# df_slope = 'slope °'

# df_main = get_data("..\\Fietssimulatie\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
# df_val = get_data("..\\Fietssimulatie\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])


df_main = get_data("..\\..\\data\\data.xlsx", [df_torque, df_crank_angle_rad, df_fcc])
df_val = get_data("..\\..\\data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_fcc])

print("Preprocessing data")
train_x, train_y = preprocess(df_main, SEQ_LEN_NN, seqs=True, normalize=True)
val_x, val_y = preprocess(df_val, SEQ_LEN_NN, shuffle=False, seqs=True, normalize=True)

print("Start learning")
name = f"SEQ_{SEQ_LEN_NN}_EPOCH_{EPOCH}_{int(time.time())}"
model = learn_lstm(name, train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCH)
predictions = model.predict(val_x)
visualize_data([predictions, val_y], ["Predicted Optimal Cadence", "FCC"])
