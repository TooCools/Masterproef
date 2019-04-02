import time

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from xsrc.analyze import visualize_data
from xsrc.bike import Bike
from xsrc.cadence_controller import CadenceController
from xsrc.params import seqlen

# file = open("C:\\Users\\Arno\\Desktop\\Masterproef\\CycleLearning\\output.txt", "a")
#
timesteps = 20000

# cadence_controller = CadenceController(
#     # PassiveAggressiveRegressor(C=1, max_iter=25, tol=0.1, warm_start=True, shuffle=True,),
#     DecisionTreeRegressor(max_depth=4),
#     ptype="none", seqlen=seqlen, verbose=True,stochastic=True)
# cycle = Bike(verbose=False)
# start = time.time()
# for h in range(1, timesteps):
#     t_cy, crank_angle, v_cy, slope = cycle.get_recent_data(h, seqlen)
#     if h > cadence_controller.warmup:
#         predicted_rpm = cadence_controller.predict(t_cy, crank_angle, v_cy, slope)
#         cycle.update(h, predicted_rpm)
#     else:
#         cycle.update(h)
#         predicted_rpm = 0
#
#     fcc_rpm = cycle.get_recent_fcc(h, 1)[0]
#     cadence_controller.update(h, predicted_rpm, fcc_rpm, cycle)
# end = time.time()
# visualize_data([cadence_controller.mse], [], ["Tijd", "mse"])
# print("Aantal keer getrained: "+str(cadence_controller.aantalkeer_getrained))
"""PA"""
# for loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
#     type = "PA-I" if loss == "epsilon_insensitive" else "PA-II"
#     for c in [1, 2, 5, 10]:
#         aantalkeer_getraind = []
#         tijd = []
#         for i in range(0, 10):
#             cadence_controller = CadenceController(
#                 PassiveAggressiveRegressor(C=c, max_iter=25, tol=0.1, warm_start=True, shuffle=True,
#                                            loss=loss),
#                 ptype="none", seqlen=seqlen, verbose=False)
#             cycle = Bike(verbose=False)
#             start = time.time()
#             for h in range(1, timesteps):
#                 t_cy, crank_angle, v_cy, slope = cycle.get_recent_data(h, seqlen)
#                 if h > cadence_controller.warmup:
#                     predicted_rpm = cadence_controller.predict(t_cy, crank_angle, v_cy, slope)
#                     cycle.update(h, predicted_rpm)
#                 else:
#                     cycle.update(h)
#                     predicted_rpm = 0
#
#                 fcc_rpm = cycle.get_recent_fcc(h, 1)[0]
#                 cadence_controller.update(h, predicted_rpm, fcc_rpm, cycle)
#             end = time.time()
#             tijd.append(end - start)
#             visualize_data([cadence_controller.mse], [], ["Tijd", "mse"],
#                            type + " mean squared error (c=" + str(c) + ")",
#                            "C:\\Users\\Arno\\Desktop\\Masterproef\\Images\\evaluatie\\pa\\" + str(i), True)
#             aantalkeer_getraind.append(cadence_controller.aantalkeer_getrained)
#             str1 = type + ": Aantal keer getrained=" + str(
#                 cadence_controller.aantalkeer_getrained) + " voor seqlen=" + str(seqlen
#                                                                                  ) + " en tijd: " + str(
#                 end - start) + " laatste mse: " + str(cadence_controller.mse[-1])
#             print(str1)
#             file.write(str1+"\n")
#         str2 = "==============================================================================" + type + " gemiddeld aantal keer getrained=" + str(
#             sum(aantalkeer_getraind) / 10) + " voor c=" + str(c
#                                                               ) + " en tijd: " + str(
#             sum(tijd) / 10) + "=============================================================================="
#         print(str2)
#         file.write(str2 + "\n")


sklearn.utils.parallel_backend("loky", n_jobs=1)
# file.write("@@@@@@@@@@@@@@@@@@@@@@@\n")
for depth in [3, 4, 5]:
    mses = []
    for estimators in [5, 10, 15, 20]:
        aantalkeer_getraind = []
        tijd = []
        for i in range(0, 10):
            cadence_controller = CadenceController(
                RandomForestRegressor(max_depth=depth, n_estimators=estimators,n_jobs=1),
                ptype="none", seqlen=seqlen, verbose=False)
            cycle = Bike(verbose=False)
            start = time.time()
            for h in range(1, timesteps):
                t_cy, crank_angle, v_cy, slope = cycle.get_recent_data(h, seqlen)
                if h > cadence_controller.warmup:
                    predicted_rpm = cadence_controller.predict(t_cy, crank_angle, v_cy, slope)
                    cycle.update(h, predicted_rpm)
                else:
                    cycle.update(h)
                    predicted_rpm = 0

                fcc_rpm = cycle.get_recent_fcc(h, 1)[0]
                cadence_controller.update(h, predicted_rpm, fcc_rpm, cycle)
            end = time.time()
            tijd.append(end - start)
            if i == 0:
                mses.append(cadence_controller.mse)
            aantalkeer_getraind.append(cadence_controller.aantalkeer_getrained)
            str1 = "RF: Aantal keer getrained=" + str(cadence_controller.aantalkeer_getrained) + " voor diepte=" + str(
                depth) + ", estimators: " + str(estimators) + " en tijd: " + str(end - start) + " laatste mse: " + str(
                cadence_controller.mse[-1])
            print(str1)
            # file.write(str1+"\n")
        str2 = "===========RF diepte=" + str(depth) + " estimators=" + str(
            estimators) + " gemiddeld aantal keer getrained=" + str(
            sum(aantalkeer_getraind) / 10) + " en tijd: " + str(sum(tijd) / 10)
        print(str2)
        # file.write(str2 + "\n")
    visualize_data(mses, ["5 bomen", "10 bomen", "15 bomen", "20 bomen"], ["Tijd", "mse"],
                   "Random Forest mean squared error (diepte=" + str(depth) + ")",
                   "C:\\Users\\Arno\\Desktop\\Masterproef\\Images\\evaluatie\\rf\\njobs1", True)

# file.write("@@@@@@@@@@@@@@@@@@@@@@@\n")
# for depth in [3, 4, 5]:
#     aantalkeer_getraind = []
#     tijd = []
#     for i in range(0, 10):
#         cadence_controller = CadenceController(
#             DecisionTreeRegressor(max_depth=depth),
#             ptype="none", seqlen=seqlen, verbose=False)
#         cycle = Bike(verbose=False)
#         start = time.time()
#         for h in range(1, timesteps):
#             t_cy, crank_angle, v_cy, slope = cycle.get_recent_data(h, seqlen)
#             if h > cadence_controller.warmup:
#                 predicted_rpm = cadence_controller.predict(t_cy, crank_angle, v_cy, slope)
#                 cycle.update(h, predicted_rpm)
#             else:
#                 cycle.update(h)
#                 predicted_rpm = 0
#
#             fcc_rpm = cycle.get_recent_fcc(h, 1)[0]
#             cadence_controller.update(h, predicted_rpm, fcc_rpm, cycle)
#         end = time.time()
#         tijd.append(end - start)
#         aantalkeer_getraind.append(cadence_controller.aantalkeer_getrained)
#         str1 = "DT: Aantal keer getrained=" + str(cadence_controller.aantalkeer_getrained) + " voor diepte=" + str(
#             depth) + " en tijd: " + str(end - start) + " laatste mse: " + str(cadence_controller.mse[-1])
#         print(str1)
#         file.write(str1 + "\n")
#
#         visualize_data([cadence_controller.mse], [], ["Tijd", "mse"],
#                        "Decision Tree mean squared error (diepte=" + str(depth) + ")",
#                        "C:\\Users\\Arno\\Desktop\\Masterproef\\Images\\evaluatie\\dt\\", True)
#     str2 = "==================DT diepte=" + str(depth) + " gemiddeld aantal keer getrained=" + str(
#         sum(aantalkeer_getraind) / 10) + " en tijd: " + str(sum(tijd) / 10)
#     print(str2)
#     file.write(str2 + "\n")
#
# file.close()
