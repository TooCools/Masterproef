import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from xsrc.analyze import visualize_data
from xsrc.bike import Bike
from xsrc.cadence_controller import CadenceController
from xsrc.datastructures import biased_reservoir_sampling, sliding_window, data_structure
from xsrc.params import seqlen

# file = open("C:\\Users\\Arno\\Desktop\\Masterproef\\CycleLearning\\output.txt", "a")
from xsrc.simulation.data_to_csv import save


# timesteps = 20000
cadence_controller = CadenceController(
    # PassiveAggressiveRegressor(C=1,max_iter=25,tol=0.1,warm_start=True,shuffle=True)
    RandomForestRegressor(max_depth=4,n_estimators=20),
    # RandomForestRegressor(max_depth=4, n_estimators=10),
    training=biased_reservoir_sampling.BiasedReservoirSampling(seqlen*20)
    # training=sliding_window.SlidingWindow(seqlen * 20)
    # training=data_structure.BiasedReservoirSamlping()
    , ptype="none", seqlen=seqlen, verbose=False,
    stochastic=False)
cycle1 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 7.5 * cm_tdc))
# cycle2 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 7.5 * cm_tdc))
# cycle2.slope_offset=9999
cycle2 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 300 * cm_slope + 60))


def fiets(bicycle, cc, timesteps, title="", update=True):
    for h in range(1, timesteps):
        t_cy, crank_angle, v_cy, slope = bicycle.get_recent_data(h, seqlen)
        predicted_rpm = -1
        if h > cc.warmup:
            predicted_rpm = cc.predict(t_cy, crank_angle, v_cy, slope)
            bicycle.update(h, predicted_rpm)
        else:
            bicycle.update(h)
        fcc_rpm = bicycle.get_recent_fcc(h, 1)[0]
        if update:
            cc.update(h, predicted_rpm, fcc_rpm, bicycle)
        cc.update_fcc(h, predicted_rpm, fcc_rpm)
    cc.stats(title)
    return cc.mse[-1], (time.time() - cc.start), cc.aantalkeer_getrained

fiets(cycle1,cadence_controller,20000)
fiets(cycle2,cadence_controller,40000)

# for i in [25]:
#     print("Windowed size: " + str(i))
#     op2demodel=[]
#     for j in range(10):
#         cycle1 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 7.5 * cm_tdc))
#         cycle2 = Bike(cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 300 * cm_slope + 60))
#         cadence_controller = CadenceController(
#             RandomForestRegressor(max_depth=4, n_estimators=20),
#             # training=biased_reservoir_sampling.BiasedReservoirSampling(seqlen * i)
#             # training=data_structure.DataStructure()
#             training=sliding_window.SlidingWindow(seqlen * i)
#             , ptype="none", seqlen=seqlen, verbose=False,
#             stochastic=False)
#         mse1, time1, trained1 = fiets(cycle1, cadence_controller, 20000)
#         mse2, time2, trained2 = fiets(cycle2, cadence_controller, 40000)
#         print("Trained on new cyclers model: " + str(trained2 - trained1))
#         op2demodel.append(trained2-trained1)
#     print("Voor size: "+str(i)+" moest ik gemiddeld "+str(sum(op2demodel)/len(op2demodel))+" keer bijleren")

# fiets(cycle1, cadence_controller)
# cadence_controller.reset()
# timesteps=40000
# fiets(cycle2, cadence_controller, update=True)
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
# for depth in [3, 4, 5]:
#     aantalkeer_getraind = []
#     tijd = []
#     mses = []
#     for i in range(0, 10):
#         cadence_controller = CadenceController(
#             DecisionTreeRegressor(max_depth=depth),
#             ptype="none", seqlen=seqlen, stochastic=True, normalize=False, verbose=False)
#         cycle = Bike(verbose=False)
#         mse, duration, trainings = fiets(cycle, cadence_controller,
#                                          "Decision Tree mean squared error (depth=" + str(depth) + ")")
#         aantalkeer_getraind.append(trainings)
#         tijd.append(duration)
#         mses.append(mse)
#     print("@@@@@@@@@@@DT with depth: " + str(depth) + " @@@@@@@@@@")
#     print("Avg MSE: " + str(sum(mses) / len(mses)))
#     print("Avg time: " + str(sum(tijd) / len(tijd)))
#     print("Avg trainings: " + str(sum(aantalkeer_getraind) / len(aantalkeer_getraind)))
#     print("@@@@@@@@@@@@@@@@@@@@@")

"""RF"""
# for depth in [3, 4, 5]:
#     for estimators in [5, 10, 15, 20]:
#         aantalkeer_getraind = []
#         tijd = []
#         mses = []
#         for i in range(0, 5):
#             cadence_controller = CadenceController(
#                 RandomForestRegressor(max_depth=depth, n_estimators=estimators),
#                 ptype="none", seqlen=seqlen, stochastic=True, normalize=False, verbose=False,training=data_structure.DataStructure(9999999999))
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
