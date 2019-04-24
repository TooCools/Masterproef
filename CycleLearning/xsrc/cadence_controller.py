import random
import time
from collections import deque
from math import cos, sin

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from xsrc.analyze import visualize_data
from xsrc.params import df_fcc, df_torque, df_crank_angle_rad, df_velocity, df_slope, timestep


class CadenceController:

    def __init__(self, model, ptype="none", seqlen=50, window_size=99999999, verbose=True, normalize=False, stochastic=False):
        self.model = model
        self.warmup = 300
        self.ptype = ptype
        self.verbose = verbose
        self.training_set = deque(maxlen=seqlen * window_size)
        self.mse = []
        self.difference = []
        self.previous_opt_cadence = []
        self.previous_predictions = []
        self.actual_fcc_mse = []
        self.previous_predictions_mse = []
        self.aantalkeer_getrained = 0
        self.seqlen = seqlen
        self.stochastic = stochastic
        self.normalize = normalize
        self.start = time.time()
        self.last_trained = 0

    def predict(self, torque, cr_angle, velocity, slope_rad):
        data = {
            df_torque: torque,
            df_crank_angle_rad: cr_angle,
            df_velocity: velocity,
            df_slope: slope_rad
        }
        x = self.preprocess(data, normalize=self.normalize)
        pred = self.model.predict([x])[0]

        # if random.random() > 0.8:
        #     pred += (random.random() - 0.5) * 20
        optimal_cadence = self.postprocess(pred)
        self.previous_opt_cadence.append(optimal_cadence)
        self.previous_predictions.append(pred)
        return optimal_cadence

    def preprocess(self, data, normalize=False):
        df = pandas.DataFrame(data)
        for col in df.columns:
            if normalize and col in [df_torque, df_velocity, df_slope]:
                if col == df_torque:
                    df[col] = self.normalize_data(0, 105, df[col])
                elif col == df_velocity:
                    df[col] = self.normalize_data(0, 45, df[col])
                else:
                    df[col] = self.normalize_data(0, 0.1, df[col])
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

    def normalize_data(self, min, max, data):
        data = data.apply(lambda x: (x - min) / (max - min))
        return data

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

    def train(self, h, cycle):
        # print("I trained at timestep: " + str(h) + " with " + str(len(self.training_set)) + " amount of data")
        torque, cr_angle, velocity, slope_rad = cycle.get_recent_data(h, self.seqlen * 2)
        fcc = cycle.get_recent_fcc(h, self.seqlen)
        for i in range(0, self.seqlen):
            data = {
                df_torque: torque[i:self.seqlen + i],
                df_crank_angle_rad: cr_angle[i:self.seqlen + i],
                df_velocity: velocity[i:self.seqlen + i],
                df_slope: slope_rad[i:self.seqlen + i]
            }
            self.training_set.append([self.preprocess(data, normalize=self.normalize), fcc[i]])
        random.shuffle(self.training_set)
        training_X = []
        training_y = []
        for X, y in self.training_set:
            training_X.append(X)
            training_y.append(y)
        self.model.fit(training_X, training_y)
        self.aantalkeer_getrained += 1

    def stochastic_training(self, h, difference, cycle):
        if difference > 5 and h - self.last_trained >= 30:
            chance = (difference / 10 - 0.3) * timestep
            if random.random() < chance:
                self.last_trained = h
                self.verbose_printing("Trained with chance")
                self.train(h, cycle)

    def update(self, h, prediction, actual, cycle):
        difference = abs(prediction - actual)
        if self.seqlen < h:
            if 2 * self.seqlen < h < self.warmup and h % 30 == 0:
                self.verbose_printing("Training")
                self.train(h, cycle)
            elif h > self.warmup and h % 30 == 0 and difference > 5 and not self.stochastic:
                self.verbose_printing("Training; diff: " + str(abs(prediction - actual)))
                self.train(h, cycle)
            elif self.stochastic and h > self.warmup:
                self.stochastic_training(h, difference, cycle)

    def update_fcc(self, h, prediction, actual):
        if h > self.warmup:
            self.actual_fcc_mse.append(actual)
            self.previous_predictions_mse.append(prediction)
            error = mean_squared_error(self.actual_fcc_mse, self.previous_predictions_mse)
            self.verbose_printing("mse: " + str(error))
            self.mse.append(error)
            self.difference.append(abs(prediction - actual))

    def reset(self):
        self.mse = []
        self.actual_fcc_mse = []
        self.previous_predictions_mse = []
        self.previous_opt_cadence = []
        self.previous_predictions = []
        self.difference = []
        self.aantalkeer_getrained = 0
        self.start = time.time()

    def verbose_printing(self, string):
        if self.verbose:
            print(string)

    def stats(self, model_name):
        print("=============================")
        print("Times trained: " + str(self.aantalkeer_getrained))
        print("MSE: " + str(self.mse[-1]))
        print("Time: " + str(time.time() - self.start))
        print("Stochastic: " + str(self.stochastic))
        print("=============================")
        visualize_data([self.mse], [], ["tijd", "mse"], model_name + " (stochastic= " + str(self.stochastic) + ")")
        visualize_data([self.difference], [], ["tijd", "difference"],
                       model_name + " (stochastic= " + str(self.stochastic) + ")")
