from math import sin, cos, pi

from xsrc.analyze import visualize_data
from xsrc.excel_to_data import get_data
from xsrc.learner import learn_PA, learn_SGD, get_SGD
from xsrc.preprocessing import preprocess_dict, df_torque, df_fcc, df_crank_angle_rad, preprocess
from xsrc.simulation.cycleModel import update, fietsers_koppel
from xsrc.simulation.params import *
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

theta_crank_rad = [0.0]  # Hoek van van de trapas
theta_crank_rad2 = [0.0]  # Hoek van de trapas %2PI
omega_crank = [0.0]  # rpm van de trapas
v_fiets = [0.0]  # snelheid in m/s
t_cy = [0.0]
t_mg1 = [0.0]
t_mg2 = [0.0]
o_mg1 = [0.0]
o_mg2 = [0.0]
t_dc_array = [0.0]
t_cyclist_no_noise = [0.0]
slope_array = [0.0]
f_grav_array = [0.0]
f_fric_array = [0.0]
f_aero_array = [0.0]
fcc_array = [0.0]
t_dc_max = 60
predicted_opt_cadence_array = []
actual_fcc = []
mse_array = []
predictions = []

t_dc = 0.0
slope_rad = 0.0  # helling waarop de fiets zich bevindt
route_slots = []
total_timesteps = 100000
seqlen = 50
start = True

model = learn_PA("test", [range(seqlen * 3)], [0])
# model = learn_SGD("test", [range(seqlen * 3)], [0])
current_opt_cadence = 120

train_x = []
train_y = []

times_trained = 0


def predict(i):
    stuff = {
        df_torque: t_cy[i - seqlen:i],
        df_fcc: fcc_array[i - seqlen:i],
        df_crank_angle_rad: theta_crank_rad[i - seqlen:i],
    }
    x, y = preprocess_dict(stuff, seqlen, normalize=False, shuffle=False)
    pred = model.predict(x)[0]
    if pred > 120:
        pred = 120
    elif pred < 40:
        pred = 40
    return pred


def train(i):
    global model
    global times_trained
    global start
    stuff = {
        df_torque: t_cy[i - seqlen * 2 - 1:i],
        df_fcc: fcc_array[i - seqlen * 2 - 1:i],
        df_crank_angle_rad: theta_crank_rad[i - seqlen * 2 - 1:i],
    }
    x, y = preprocess_dict(stuff, seqlen, normalize=False, shuffle=True)
    concat_training(x, y)
    if start:
        model = learn_PA("", train_x, train_y)
        start = False
    else:
        model = model.fit(train_x, train_y)
    times_trained += 1


def concat_training(x, y):
    global train_x
    global train_y

    if len(train_x) == 0:
        train_x = x
        train_y = y
    else:
        train_x = np.concatenate((train_x, x))
        train_y = train_y + y


mse_fcc = []
mse_model_predictions = []
accuracy_predicted_opt_cadense = []
accuracy_fcc = []
warmup = 300


def score(i, model_prediction, predicted_opt_cadence, fcc, diff):
    if i > warmup:
        if predicted_opt_cadence == 40 and fcc == 40:
            print("Predicted: " + str(predicted_opt_cadence), "Actual: " + str(fcc_array[h]),
                  "Difference: " + str(diff) + " T: " + str(i))
        else:
            mse_fcc.append(fcc)
            accuracy_predicted_opt_cadense.append(predicted_opt_cadence)
            mse_model_predictions.append(model_prediction)
            mse_value = mean_squared_error(mse_fcc, mse_model_predictions)
            accuracy = accuracy_score(mse_fcc, accuracy_predicted_opt_cadense)
            mse_array.append(mse_value)
            accuracy_fcc.append(accuracy)
            print("Predicted: " + str(predicted_opt_cadence), "Actual: " + str(fcc_array[h]),
                  "Difference: " + str(diff),
                  "MSE: " + str(mse_value) + " Accuracy: " + str(accuracy) + " T: " + str(i))


def bicycle_model():
    avg_tdc = np.average(t_dc_array[-50:])
    # fcc = 7 * t_dc_array[-1:][0] - 10
    fcc = 7 * avg_tdc - 10
    fcc = int(5 * round(fcc / 5))
    if fcc < 40:
        fcc = 40
    elif fcc > 120:
        fcc = 120
    return fcc


smoothing_factor = 0.1


def postprocessing(model_prediction):
    # if model_prediction < 40:
    #     model_prediction = 40
    # elif model_prediction > 120:
    #     model_prediction = 120
    # predicted_opt_cadence = int(5 * round(model_prediction / 5))
    # return predicted_opt_cadence
    # previous_predictions = predictions[-5:]
    # return int(5 * round(np.average(previous_predictions) / 5))
    if len(mse_model_predictions) == 0:
        return int(5 * round(model_prediction / 5))
    else:
        previous_pred = mse_model_predictions[-1]
        smoothed = smoothing_factor * previous_pred + (1 - smoothing_factor) * model_prediction
        return int(5 * round(smoothed / 5))


def machine_learning():
    global current_opt_cadence
    if h > seqlen:
        model_prediction = predict(h)
        predictions.append(model_prediction)
        if h % 5 == 0:
            predicted_opt_cadence = postprocessing(model_prediction)
            predicted_opt_cadence_array.append(predicted_opt_cadence)
            diff = bicycle_model() - predicted_opt_cadence
            score(h, model_prediction, predicted_opt_cadence, fcc_array[h], diff)
            actual_fcc.append(fcc_array[h])
            if h % 30 == 0 and h > 2 * seqlen:
                if abs(diff) > 5:
                    print("Training the model because the difference was too high; diff= " + str(diff))
                    train(h)
            if h >= warmup:
                current_opt_cadence = predicted_opt_cadence
    if h < warmup:
        current_opt_cadence = bicycle_model()


def cadence_for_speed(v):
    if v <= 15:
        rpm = v * 70 / 15
    else:
        rpm = 70 + (v - 15) * 8
    if rpm > current_opt_cadence:
        return current_opt_cadence
    return rpm


for h in range(1, int(total_timesteps)):
    a, b, c = update(h, omega_crank, v_fiets)
    t_dc = a
    t_dc_max = b
    slope_rad = c
    v_fiets_previous_kmh = v_fiets[h - 1]
    v_fiets_previous_ms = v_fiets_previous_kmh / 3.6

    omega_crank_current_rpm = cadence_for_speed(
        v_fiets_previous_kmh)  # min(omega_opt_rpm, cadence_for_speed(v_fiets_previous_kmh)) + rpm_offset
    omega_crank_current_rads = omega_crank_current_rpm * 0.10467

    theta_crank_current_rad = theta_crank_rad[h - 1] + omega_crank_current_rads * timestep

    # Dit zijn waarden voor het vermogen (torque) van de fietser + motor generatoren op het voor- (2) en achterwiel (1)
    t_cyclist = fietsers_koppel(theta_crank_current_rad, t_dc, t_dc_array, t_cyclist_no_noise, dominant_leg=True)
    t_mg1_current = t_cyclist * kcr_r * (ns / nr) * ks_mg1
    t_mg2_current = min(20, support_level * t_cyclist)
    t_rw = t_cyclist * kcr_r * ((nr + ns) / nr)

    f_grav = total_mass * g * sin(slope_rad) * 0.9
    f_friction = total_mass * g * cos(slope_rad) * cr
    f_aero = 0.5 * cd * ro_aero * a_aero * (v_fiets_previous_ms ** 2)
    f_aero *= np.sign(v_fiets_previous_kmh)
    f_load = f_grav + f_friction + f_aero

    v_fiets_next_ms = (((t_mg2_current + t_rw) / rw) - f_load) / total_mass
    v_fiets_current_ms = v_fiets_previous_ms + timestep * v_fiets_next_ms

    omega_mg2 = v_fiets_current_ms / rw
    omega_mg1 = (1 / ks_mg1) * ((1 + (nr / ns)) * omega_mg2 - ((nr / ns) * (omega_crank_current_rads / kcr_r)))

    theta_crank_rad.append(theta_crank_current_rad)
    theta_crank_rad2.append(theta_crank_current_rad % (2 * pi))
    omega_crank.append(omega_crank_current_rpm)
    v_fiets.append(v_fiets_current_ms * 3.6)
    t_cy.append(t_cyclist)
    t_mg1.append(t_mg1_current)
    t_mg2.append(t_mg2_current)
    o_mg1.append(omega_mg1)
    o_mg2.append(omega_mg2)
    slope_array.append(slope_rad * 57.296)
    f_grav_array.append(f_grav)
    f_fric_array.append(f_friction)
    f_aero_array.append(f_aero)
    # print('time', int(h / 10), 'speed', v_fiets_previous_kmh, 'slope', slope_rad, 'rpm', omega_crank_current_rpm,
    #       'tdc',
    #       t_dc_array[h])

    fcc_array.append(bicycle_model())
    machine_learning()

data = {'speed (km/h)': v_fiets,
        'rpm': omega_crank,
        'crank_angle': theta_crank_rad,
        'crank_angle_%2PI': theta_crank_rad2,
        't_dc': t_dc_array,
        't_cyclist': t_cy,
        't_cyclist_no_noise': t_cyclist_no_noise,
        't_mg1': t_mg1,
        't_mg2': t_mg2,
        'o_mg1': o_mg1,
        'o_mg2': o_mg2,
        'slope °': slope_array,
        'force gravity': f_grav_array,
        'force friction': f_fric_array,
        'force aero': f_aero_array,
        'fcc': fcc_array
        }

visualize_data([predicted_opt_cadence_array, actual_fcc], ["Predicted Optimal Cadence", "FCC"])
visualize_data([mse_array], ["MSE"])
visualize_data([accuracy_fcc], ["Accuracy"])

print("Times trained: " + str(times_trained))

# save(data)
# save(data, "validation")

# visualize_data([omega_crank, slope_array])

# def visualize_data(y):
#     for data in y:
#         plt.plot(data)
#     plt.show()
