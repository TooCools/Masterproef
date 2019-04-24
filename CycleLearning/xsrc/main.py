import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from xsrc.analyze import visualize_data
from xsrc.bike import Bike
from xsrc.cadence_controller import CadenceController
from xsrc.params import seqlen

# file = open("C:\\Users\\Arno\\Desktop\\Masterproef\\CycleLearning\\output.txt", "a")
from xsrc.simulation.data_to_csv import save

timesteps = 20000


# cadence_controller = CadenceController(
#     # PassiveAggressiveRegressor(C=1,max_iter=25,tol=0.1,warm_start=True,shuffle=True)
#     RandomForestRegressor(max_depth=4, n_estimators=10), window_size=10
#     , ptype="none", seqlen=seqlen, verbose=False, stochastic=False)
# cycle1 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 7.5 * cm_tdc))
# cycle2 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 300 * cm_slope + 60))


def fiets(b, cc, title, update=True):
    for h in range(1, timesteps):
        t_cy, crank_angle, v_cy, slope = b.get_recent_data(h, seqlen)
        predicted_rpm = -1
        if h > cc.warmup:
            predicted_rpm = cc.predict(t_cy, crank_angle, v_cy, slope)
            b.update(h, predicted_rpm)
        else:
            b.update(h)
        fcc_rpm = b.get_recent_fcc(h, 1)[0]
        if update:
            cc.update(h, predicted_rpm, fcc_rpm, b)
        cc.update_fcc(h, predicted_rpm, fcc_rpm)
    cc.stats(title)
    return cc.mse[-1], (time.time() - cc.start), cc.aantalkeer_getrained


# fiets(cycle1, cadence_controller)
# cadence_controller.reset()
# fiets(cycle2, cadence_controller,update=True)
# feature_names=[]
# for i in range(50):
#     feature_names.append("torque"+str(i))
#     feature_names.append("speed" + str(i))
#     feature_names.append("slope" + str(i))
#     feature_names.append("cos"+str(i))
#     feature_names.append("sin" + str(i))
#
# from sklearn.tree import export_graphviz
# export_graphviz(cadence_controller.model.estimators_[5], out_file='slope_model.dot',
#                 feature_names = feature_names,
#                 rounded = True, proportion = False,
#                 precision = 2, filled = True)

"""PA"""
# for loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
#     type = "PA-I" if loss == "epsilon_insensitive" else "PA-II"
#     for c in [1, 2, 5, 10]:
#         aantalkeer_getraind = []
#         tijd = []
#         mses = []
#         for i in range(0, 10):
#             cadence_controller = CadenceController(
#                 PassiveAggressiveRegressor(C=c, max_iter=25, tol=0.1, warm_start=True, shuffle=False,
#                                            loss=loss),
#                 ptype="none", seqlen=seqlen, stochastic=True, normalize=False, verbose=False)
#             cycle = Bike(verbose=False)
#             mse, duration, trainings = fiets(cycle, cadence_controller, type + " mean squared error (c=" + str(c) + ")")
#             aantalkeer_getraind.append(trainings)
#             tijd.append(duration)
#             mses.append(mse)
#         print("@@@@@@@@@@" + type + " with c=" + str(c) + "@@@@@@@@@@@")
#         print("Avg MSE: " + str(sum(mses) / len(mses)))
#         print("Avg time: " + str(sum(tijd) / len(tijd)))
#         print("Avg trainings: " + str(sum(aantalkeer_getraind) / len(aantalkeer_getraind)))
#         print("@@@@@@@@@@@@@@@@@@@@@")

"""DT"""
for depth in [3, 4, 5]:
    aantalkeer_getraind = []
    tijd = []
    mses = []
    for i in range(0, 10):
        cadence_controller = CadenceController(
            DecisionTreeRegressor(max_depth=depth),
            ptype="none", seqlen=seqlen, stochastic=True, normalize=False, verbose=False)
        cycle = Bike(verbose=False)
        mse, duration, trainings = fiets(cycle, cadence_controller,
                                         "Decision Tree mean squared error (depth=" + str(depth) + ")")
        aantalkeer_getraind.append(trainings)
        tijd.append(duration)
        mses.append(mse)
    print("@@@@@@@@@@@DT with depth: " + str(depth) + " @@@@@@@@@@")
    print("Avg MSE: " + str(sum(mses) / len(mses)))
    print("Avg time: " + str(sum(tijd) / len(tijd)))
    print("Avg trainings: " + str(sum(aantalkeer_getraind) / len(aantalkeer_getraind)))
    print("@@@@@@@@@@@@@@@@@@@@@")

"""RF"""
# for depth in [3, 4, 5]:
#     for estimators in [5, 10, 15, 20]:
#         aantalkeer_getraind = []
#         tijd = []
#         mses = []
#         for i in range(0, 10):
#             cadence_controller = CadenceController(
#                 RandomForestRegressor(max_depth=depth, n_estimators=estimators),
#                 ptype="none", seqlen=seqlen, stochastic=True, normalize=False, verbose=False)
#             cycle = Bike(verbose=False)
#             mse, duration, trainings = fiets(cycle, cadence_controller,
#                                              "Decision Tree mean squared error (depth=" + str(depth) + ")")
#             aantalkeer_getraind.append(trainings)
#             tijd.append(duration)
#             mses.append(mse)
#         print("@@@@@@@@@@@RF with depth: " + str(depth) + " and estimators: " + str(estimators) + " @@@@@@@@@@")
#         print("Avg MSE: " + str(sum(mses) / len(mses)))
#         print("Avg time: " + str(sum(tijd) / len(tijd)))
#         print("Avg trainings: " + str(sum(aantalkeer_getraind) / len(aantalkeer_getraind)))
#         print("@@@@@@@@@@@@@@@@@@@@@")
