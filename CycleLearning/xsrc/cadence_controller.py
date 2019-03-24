import random
from math import cos, sin

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Dropout, BatchNormalization, Dense
from tensorflow.python.keras.optimizers import Adam

from xsrc.params import seqlen, df_fcc, df_torque, df_crank_angle_rad, df_velocity, df_slope
from xsrc.stuff.preprocessing import preprocess_keras, preprocess_keras2


class CadenceController:

    def __init__(self, model, ptype="none", verbose=True, keras=False):
        self.model = model
        self.warmup = 300
        self.ptype = ptype
        self.verbose = verbose
        self.training_set = []
        self.mse = []
        self.previous_opt_cadence = []
        self.previous_predictions = []
        self.actual_fcc_mse = []
        self.previous_predictions_mse = []
        self.aantalkeer_getrained = 0
        self.keras = keras

    def predict(self, torque, cr_angle, velocity, slope_rad):
        data = {
            df_torque: torque,
            df_crank_angle_rad: cr_angle,
            df_velocity: velocity,
            df_slope: slope_rad
        }
        if not self.keras:
            x = self.preprocess(data, normalize=False)
            pred = self.model.predict([x])[0]
        else:
            x, y = preprocess_keras2(data, np.zeros(50), seqlen, normalize=False, shuffle=True, seqs=True)
            pred = self.model.predict(x)[0][0]
        # if random.random() > 0.8:
        #     pred += (random.random() - 0.5) * 20
        optimal_cadence = self.postprocess(pred)
        self.previous_opt_cadence.append(optimal_cadence)
        self.previous_predictions.append(pred)
        return optimal_cadence

    def preprocess(self, data, normalize=False):
        df = pandas.DataFrame(data)
        min_max_scaler = preprocessing.MinMaxScaler()
        for col in df.columns:
            if normalize and col in [df_torque, df_velocity, df_slope]:
                vals_scaled = min_max_scaler.fit_transform(df[[col]])
                df_new = pandas.DataFrame(vals_scaled)
                df[col] = df_new
            if col == df_crank_angle_rad:
                cosvals = []
                sinvals = []
                for val in df[col].values:
                    cosvals.append(cos(val))
                    sinvals.append(sin(val))
                df['cos_angle'] = pandas.Series(cosvals)
                df['sin_angle'] = pandas.Series(sinvals)
        df = df.drop(df_crank_angle_rad, 1)
        x = []
        for row in df.values.tolist():
            for val in row:
                x.append(val)
        return x

    def postprocess(self, prediction):
        if prediction < 40:
            prediction = 40
        elif prediction > 120:
            prediction = 120

        if self.ptype == "ma":
            p = self.previous_opt_cadence[-9:]
            p.append(prediction)
            return np.average(p)
        elif self.ptype == "es":
            sf = 0.5
            if len(self.previous_opt_cadence) == 0:
                return prediction
            return sf * self.previous_opt_cadence[-1] + (1 - sf) * prediction
        elif self.ptype == "combo":
            p = self.previous_predictions[-9:]
            p.append(prediction)
            curr = np.average(p)
            sf = 0.5
            if len(self.previous_opt_cadence) == 0:
                return curr
            return sf * self.previous_opt_cadence[-1] + (1 - sf) * curr
        return prediction

    def train_keras(self, torque, cr_angle, velocity, slope_rad, fcc):
        data = {
            df_torque: torque,
            df_crank_angle_rad: cr_angle,
            df_velocity: velocity,
            df_slope: slope_rad
        }
        X, y = preprocess_keras2(data, fcc, seqlen, normalize=False, seqs=True)
        for i in range(len(X)):
            self.training_set.append([X[i], y[i]])
        random.shuffle(self.training_set)
        training_X = []
        training_y = []
        for X, y in self.training_set:
            training_X.append(X)
            training_y.append(y)
        self.model.fit(np.array(training_X), np.array(training_y),epochs=5,verbose=0)
        # self.model.fit(X,ys)
        # print("noice")

    def train(self, h, cycle):
        torque, cr_angle, velocity, slope_rad = cycle.get_recent_data(h, seqlen * 2)
        fcc = cycle.get_recent_fcc(h, seqlen)

        if self.keras:
            self.train_keras(torque, cr_angle, velocity, slope_rad, fcc)
        else:
            for i in range(0, seqlen):
                data = {
                    df_torque: torque[i:seqlen + i],
                    df_crank_angle_rad: cr_angle[i:seqlen + i],
                    df_velocity: velocity[i:seqlen + i],
                    df_slope: slope_rad[i:seqlen + i]
                }
                self.training_set.append([self.preprocess(data), fcc[i]])
            random.shuffle(self.training_set)
            training_X = []
            training_y = []
            for X, y in self.training_set:
                training_X.append(X)
                training_y.append(y)
            self.model.fit(training_X, training_y)
        self.aantalkeer_getrained += 1

    def update(self, h, prediction, actual, cycle):
        if seqlen < h:
            if 2 * seqlen < h < self.warmup and h % 30 == 0:
                self.verbose_printing("Training")
                self.train(h, cycle)
            elif h > self.warmup and h % 30 == 0 and abs(prediction - actual) > 5:
                self.verbose_printing("Training; diff: " + str(abs(prediction - actual)))
                self.train(h, cycle)
            if h > self.warmup:
                self.actual_fcc_mse.append(actual)
                self.previous_predictions_mse.append(prediction)
                error = mean_squared_error(self.actual_fcc_mse, self.previous_predictions_mse)
                self.verbose_printing("mse: " + str(error))
                self.mse.append(error)

    def verbose_printing(self, string):
        if self.verbose:
            print(string)
