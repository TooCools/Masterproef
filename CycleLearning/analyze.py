import tensorflow

from excel_to_data import get_data
import matplotlib.pyplot as plt
from preprocessing import preprocess

df_torque = 't_cyclist'
df_crank_angle_rad = 'crank_angle_%2PI'
df_rpm = 'rpm'
df_val = get_data("data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])

val_x, val_y = preprocess(df_val, False)

model=tensorflow.keras.models.load_model("models/RNN_Final-02-66.951.model")
predictions = model.predict(val_x)
plt.figure()
plt.plot(range(0,len(val_y)),val_y)
plt.plot(range(0,len(predictions)),predictions)
plt.show()
