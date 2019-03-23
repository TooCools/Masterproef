from math import pi

import pandas as pd
import matplotlib.pyplot as plt


# visualize_data(torque[index:],["Human torque"])

def fix_crankangle(crankangle):
    a = []
    previous = 0
    timespi = 0
    for val in crankangle:
        if previous > val:
            timespi += 1
        previous = val
        a.append(2 * timespi * pi + val)
    return a


df = pd.read_excel("..\\..\\..\\Data\\BikeControl.xlsx")
torque = df[" Crank_torque_Nm_filt"]
cr_angle = df[" Crankangle"]
rads = df[" Crank_speed_rads"]

index = 7898
offset=25
stuff = torque[index:index + offset]
crankangle = cr_angle[index:index + offset]
rads = rads[index:index + offset]

avgrads=pd.np.average(rads)

rpm=avgrads*9.549297
print(rpm)

print(crankangle.head())
a = fix_crankangle(crankangle)
plt.plot(a,stuff.reset_index()[" Crank_torque_Nm_filt"])
plt.xlabel("Hoek (rad)")
plt.ylabel("Koppel (Nm)")
# plt.legend(["Menselijk koppel"])
plt.show()

