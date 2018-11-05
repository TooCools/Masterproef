from excel_to_data import get_data

from learner import learn_lstm
from params import *
from preprocessing import preprocess
import time

# df_slope = 'slope °'

# df_main = get_data("..\\Fietssimulatie\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
# df_val = get_data("..\\Fietssimulatie\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])

print("Loading Data")
df_main = get_data("data\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
df_val = get_data("data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
print("Data loaded")

print("Preprocessing data")
train_x, train_y = preprocess(df_main, SEQ_LEN_NN)
val_x, val_y = preprocess(df_val, SEQ_LEN_NN, shuffle=False)

print("Start learning")
name = f"SEQ_{SEQ_LEN_NN}_EPOCH_{EPOCH}_{int(time.time())}"
model = learn_lstm(name, train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCH)
