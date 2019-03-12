import pandas as pd
import matplotlib.pyplot as plt
from xsrc.analyze import visualize_data

df = pd.read_excel("..\\..\\..\\Data\\BikeControl.xlsx")
print(df.head())
torque=df[" Crank_torque_Nm_filt"]
index=7900
print(torque[index:])
stuff=torque[index:index+50]
print()
plt.plot(stuff.reset_index()[" Crank_torque_Nm_filt"])
plt.xlabel("Tijd")
plt.ylabel("Koppel (Nm)")
# plt.legend(["Menselijk koppel"])
plt.show()

# visualize_data(torque[index:],["Human torque"])
