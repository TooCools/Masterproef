import tensorflow

from excel_to_data import get_data
import matplotlib.pyplot as plt
from preprocessing import preprocess
from params import *
import pickle
import numpy
from collections import deque

df_torque = 't_cyclist'
df_crank_angle_rad = 'crank_angle_%2PI'
df_rpm = 'rpm'
df_val = get_data("data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])

val_x, val_y = preprocess(df_val, SEQ_LEN_TREE, normalize=False, shuffle=False)
# val_x, val_y = preprocess(df_val, SEQ_LEN_NN, normalize=True, shuffle=False)

# model=tensorflow.keras.models.load_model("models/RNN_Final-02-66.951.model")
# predictions = model.predict(val_x)
# size = 300
# val_y_s = val_y[:size]
# predictions_s = predictions[:size]
# plt.figure()
# plt.plot(range(0, len(val_y_s)), val_y_s)
# plt.plot(range(0, len(predictions_s)), predictions_s, 'ro', markersize=1)#todo check dit is (hoe een groot verschil er zit)
# plt.show()


model = pickle.load(open("models/forest.sav", "rb"))
nsamples, nx, ny = val_x.shape
d2_val_dataset = val_x.reshape((nsamples, nx * ny))
predictions = model.predict(d2_val_dataset)
size = 300
val_y_s = val_y[:size]
predictions_s = predictions[:size]
plt.figure()
plt.plot(range(0, len(val_y_s)), val_y_s)
plt.plot(range(0, len(predictions_s)), predictions_s, 'ro', markersize=1)
plt.show()

q = deque(maxlen=5)
avg_prediction = []
for pred in predictions_s:
    q.append(pred)
    if len(q) == 5:
        avg_prediction.append(numpy.mean(q))
    else:
        avg_prediction.append(pred)

plt.figure()
plt.plot(range(0, len(val_y_s)), val_y_s)
plt.plot(range(0, len(avg_prediction)), avg_prediction, 'ro', markersize=1)
plt.show()
