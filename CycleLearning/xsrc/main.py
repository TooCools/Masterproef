from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import CuDNNLSTM, Dropout, BatchNormalization, Dense, LSTM
from tensorflow.python.keras.optimizers import Adam

from xsrc.analyze import visualize_data
from xsrc.bike import Bike
from xsrc.cadence_controller import CadenceController
from xsrc.params import seqlen

timesteps = 5000
"""PA"""
# for loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
#     for c in [1, 2, 5, 10]:
#         aantalkeer_getraind = []
#         for i in range(0, 10):
#             cadence_controller = CadenceController(
#                 PassiveAggressiveRegressor(C=c, max_iter=25, tol=0.1, warm_start=True, shuffle=True,
#                                            loss=loss), ptype="none", verbose=False)
#             cycle = Bike(verbose=False)
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
#             type = "PA-I" if loss == "epsilon_insensitive" else "PA-II"
#             visualize_data([cadence_controller.mse], [], ["Tijd", "mse"],
#                            type + " mean squared error (c=" + str(c) + ")",
#                            "C:\\Users\\Arno\\Desktop\\Masterproef\\Images\\evaluatie\\pa\\" + str(i), True)
#             aantalkeer_getraind.append(cadence_controller.aantalkeer_getrained)
#         print("Gemiddeld aantal keer getrained=" + str(sum(aantalkeer_getraind) / 10) + " voor c=" + str(
#             c) + " en loss: " + loss+" alle trainingdinge = "+str(aantalkeer_getraind))

"""DT en RF"""
# for model in ["DT"]:
#     for depth in [3, 4, 5]:
#         # data = [[],[],[],[],[],[],[],[],[],[]]
#         # legende = []
#         # for nestimators in [5, 10,15,20]:
#         aantalkeer_getraind = []
#         # legende.append(str(nestimators)+" bomen")
#         for i in range(0, 10):
#             if model == "RF":
#                 cadence_controller = CadenceController(
#                     RandomForestRegressor(max_depth=depth, n_estimators=10), ptype="none",
#                     verbose=False)
#             else:
#                 cadence_controller = CadenceController(
#                     DecisionTreeRegressor(max_depth=depth), ptype="none", verbose=False)
#             cycle = Bike(verbose=False)
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
#             aantalkeer_getraind.append(cadence_controller.aantalkeer_getrained)
#             visualize_data([cadence_controller.mse], [], ["Tijd", "mse"],
#                            "Decision Tree mean squared error (diepte=" + str(depth) + ")",
#                            "C:\\Users\\Arno\\Desktop\\Masterproef\\Images\\evaluatie\\dt\\" + str(i), True)
#         print("Gemiddeld aantal keer getrained=" + str(sum(aantalkeer_getraind) / 10) + " voor depth=" + str(
#             depth) + " alle trainingdinge = " + str(aantalkeer_getraind))

# legende.append(str(nestimators)+" bomen")
# visualize_data(data, [], ["Tijd", "mse"],
#                "Random Forest mean squared error (diepte=" + str(depth) + ")",
#                "C:\\Users\\Arno\\Desktop\\Masterproef\\Images\\evaluatie\\20000", True)


model = Sequential()
# model.add(LSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))
opt = Adam(lr=5, decay=0)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
    metrics=['mean_squared_error']
)
cadence_controller = CadenceController(model
                                       , ptype="none", verbose=True,keras=True)
cycle = Bike(verbose=False)
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
