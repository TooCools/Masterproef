import random
import time
from math import cos, sin

import pandas
import numpy as np
from sklearn.metrics import mean_squared_error

from analyze import visualize_data
from params import df_torque, df_crank_angle_rad, df_velocity, df_slope, timestep


class CadenceController:

    def __init__(self, model, data_structure, ptype="none", seqlen=50,
                 verbose=True,
                 normalize=False,
                 stochastic=False):
        '''
        Creates a cadence controller that learns the preferred cadence of a cycler based on feedback of said cycler
        :param model: model used to learn preferred cadence
        :param data_structure: which structure hold the data (should be of class data_structure.DataStructure)
        :param ptype: what type of postprocessing to use ("ma" or "es")
        :param seqlen: size of sequences
        :param verbose: boolean,whether to print extra data or not to the console
        :param normalize: boolean to normalize the data or not
        :param stochastic: boolean,whether the cycle uses a determenistic update strategy or not
        '''
        self.model = model
        self.warmup = 300  # warmup period / used because we cannot predict when nothing is learned
        self.ptype = ptype
        self.verbose = verbose
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
        self.data_structure = data_structure

    def predict(self, torque, cr_angle, velocity, slope_rad):
        '''
        Generates a prediction based on 4 parameters, each being an array of #seqlen values
        :param torque: the measured cyclers torque
        :param cr_angle: the measured crank angle
        :param velocity: the measured speed
        :param slope_rad: the slope of the bike
        :return: a prediction for the preferred cadence
        '''
        data = {
            df_torque: torque,
            df_crank_angle_rad: cr_angle,
            df_velocity: velocity,
            df_slope: slope_rad
        }
        x = self.preprocess(data, normalize=self.normalize)
        pred = self.model.predict([x])[0]

        optimal_cadence = self.postprocess(pred)
        self.previous_opt_cadence.append(optimal_cadence)
        self.previous_predictions.append(pred)
        return optimal_cadence

    def preprocess(self, data, normalize=False):
        '''
        Preprocesses the data, sin + cos of crank angle, normalize if needed
        :param data: data to preprocess (a default python dictionary)
        :param normalize: boolean to chose to normalize the data or not
        :return:
        '''
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
        '''
        Does a min-max normalization, min and max should be predefined since it only checks the current data used for
        prediction. If this is not done, the normalization would have errors (imagine starting the bike => high torque
        needed <=> when driving comfortably, the torque will be lower, taking max(data) and then normalizing will
        generate wrong data
        :param min: minimum value
        :param max: maximum value
        :param data: data to normalize
        :return: normalized data
        '''
        data = data.apply(lambda x: (x - min) / (max - min))
        return data

    def postprocess(self, prediction):
        '''
        Limits the prediction between 40 and 120. If prediction is set to ma or es, applies chosen smoothing technique
        :param prediction: current prediction
        :return: cadence setting to use
        '''
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
        return prediction

    def train(self, h, cycle):
        '''
        Train the model once and deterministically, based on data of the last 5 seconds
        :param h: current iteration
        :param cycle: the cycle used
        :return:
        '''
        torque, cr_angle, velocity, slope_rad = cycle.get_recent_data(h, self.seqlen * 2)
        fcc = cycle.get_recent_fcc(h, self.seqlen)
        for i in range(0, self.seqlen):
            data = {
                df_torque: torque[i:self.seqlen + i],
                df_crank_angle_rad: cr_angle[i:self.seqlen + i],
                df_velocity: velocity[i:self.seqlen + i],
                df_slope: slope_rad[i:self.seqlen + i]
            }
            self.data_structure.add_element([self.preprocess(data, normalize=self.normalize), fcc[i]])
        training_X = []
        training_y = []
        for X, y in self.data_structure.get_elements():
            training_X.append(X)
            training_y.append(y)
        self.model.fit(training_X, training_y)
        self.aantalkeer_getrained += 1

    def stochastic_training(self, h, difference, cycle):
        '''
        Stochastically updates the model
        :param h: current iteration
        :param difference: the absolute difference between prediction and FCC
        :param cycle: the cycle used
        :return:
        '''
        if h - self.last_trained >= 30:
            if difference > 5:
                chance = (difference * 0.16 - 0.6) * timestep  # lineair van 0.2-1
                if chance > 1 * timestep:
                    chance = 1 * timestep
            else:
                chance = difference * 0.04 * timestep  # lineair van 0-0.2
            if random.random() < chance:
                self.last_trained = h
                self.verbose_printing("Trained with chance")
                self.train(h, cycle)

    def update(self, h, prediction, actual, cycle):
        '''
        Updates the cadence controller. This chooses whether to update or not. Updates are at least 30 iterations spread
        apart.
        :param h: current iteration
        :param prediction: prediction made
        :param actual: FCC
        :param cycle: cycle used
        :return:
        '''
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

    def update_error(self, h, prediction, actual):
        '''
        Keeps track of the mean squared error made
        :param h: current iteration
        :param prediction: predicted cadence
        :param actual: actual cadence (FCC)
        :return:
        '''
        if h > self.warmup:
            self.actual_fcc_mse.append(actual)
            self.previous_predictions_mse.append(prediction)
            error = mean_squared_error(self.actual_fcc_mse, self.previous_predictions_mse)
            self.verbose_printing("mse: " + str(error))
            self.mse.append(error)
            self.difference.append(abs(prediction - actual))


    def reset(self):
        '''
        Resets the cadence controller data, except mse,controller
        :return:
        '''
        # self.mse = []
        self.actual_fcc_mse = []
        self.previous_predictions_mse = []
        self.previous_opt_cadence = []
        self.previous_predictions = []
        self.difference = []
        self.aantalkeer_getrained = 0
        self.start = time.time()

    def verbose_printing(self, string):
        '''
        Prints a string only if verbose is set to true
        :param string: string to print
        :return:
        '''
        if self.verbose:
            print(string)


    def stats(self):
        '''
        Prints statistics and visualizes the MSE and difference (|pred-FCC|)
        :return:
        '''
        print("=============================")
        print("Times trained: " + str(self.aantalkeer_getrained))
        print("MSE: " + str(self.mse[-1]))
        print("Time: " + str(time.time() - self.start))
        print("Stochastic: " + str(self.stochastic))
        print("=============================")
        visualize_data([self.mse], [], ["tijd", "mse"])
        visualize_data([self.difference], [], ["tijd", "absoluut verschil voorspelling en fcc"])
