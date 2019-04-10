from math import pi

import pandas as pd
import matplotlib.pyplot as plt


# def fix_crankangle(crankangle):
#     a = []
#     previous = 0
#     timespi = 0
#     for val in crankangle:
#         if previous > val:
#             timespi += 1
#         previous = val
#         a.append(2 * timespi * pi + val)
#     return a
#
#




df = pd.read_excel("..\\..\\..\\Data\\BikeControl.xlsx")
torque = df[" Crank_torque_Nm_filt"]
print(torque.tail())
torque_norm = normalize(0, 120, torque)
print(torque_norm.tail())
torque = normalize(0, 120, torque)
print(torque.tail())

#
# index = 7898
# offset=25
# stuff = torque[index:index + offset]
# crankangle = cr_angle[index:index + offset]
# rads = rads[index:index + offset]
#
# avgrads=pd.np.average(rads)
#
# rpm=avgrads*9.549297
# print(rpm)
#
# print(crankangle.head())
# a = fix_crankangle(crankangle)
# plt.plot(a,stuff.reset_index()[" Crank_torque_Nm_filt"])
# plt.xlabel("Hoek (rad)")
# plt.ylabel("Koppel (Nm)")
# # plt.legend(["Menselijk koppel"])
# plt.show()

# x=[]
# y_min=[]
# y_max=[]
# for i in range(0,50):
#     x.append(i)
#     if i>30:
#         y_max.append(120)
#     else:
#         y_max.append(i*4)
#     if i>20:
#         y_min.append(40)
#     else:
#         y_min.append(i*2)
#
# plt.plot(x,y_min)
# plt.plot(x,y_max)
# plt.xlabel("Snelheid (km/h)")
# plt.ylabel("Cadans (rpm)")
# plt.legend(["Minimum cadans","Maximum cadans"])
# plt.xlim([0,50])
# plt.ylim([0,130])
# plt.show()
