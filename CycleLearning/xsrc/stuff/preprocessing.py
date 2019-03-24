from math import sin, cos

from sklearn import preprocessing
import pandas
from collections import deque
import numpy as np
from xsrc.stuff.params import *
import random


def preprocess(df, seq_len, normalize=True, shuffle=True, classification=False, seqs=False):
    '''
    Scales the torque column with a minmax scaler
    Add 2 new rows based on crank angle, removes crank angle column
    Creates sequences of length SEQ_LEN
    :param df: The dataframe
    :return: x: the sequenced data
             y: the resulting data
    '''
    # print("Preprocessing")
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in df.columns:
        if normalize and col != df_fcc and col != df_crank_angle_rad:
            vals_scaled = min_max_scaler.fit_transform(df[[col]])
            df_new = pandas.DataFrame(vals_scaled)
            df[col] = df_new
            # df[col] = preprocessing.scale(df[col].values) todo check if this is better
        if col == df_crank_angle_rad:
            cosvals1 = []
            sinvals1 = []
            for val in df[col].values:
                cosvals1.append(cos(val))
                sinvals1.append(sin(val))
            df['cos_angle'] = pandas.Series(cosvals1)
            df['sin_angle'] = pandas.Series(sinvals1)

    if classification:
        df[df_fcc] = (df[df_fcc] - 40) / 5
        df[df_fcc][df[df_fcc] < 0] = 0

    df = df.drop(df_crank_angle_rad, 1)

    df = df[[df_fcc, df_torque, 'cos_angle', 'sin_angle']]
    sequences = []
    prev_data = deque(maxlen=seq_len)
    for i in df.values:
        prev_data.append([n for n in i[1:]])
        if len(prev_data) == seq_len:
            sequences.append([np.array(prev_data), i[0]])
    if shuffle:
        random.shuffle(sequences)
    x = []
    y = []
    for seq, target in sequences:
        x.append(seq)
        y.append(target)
    # print("Preprocessing Finished")
    xs = np.array(x)
    if not seqs:
        nsamples, nx, ny = xs.shape
        return xs.reshape((nsamples, nx * ny)), y
    return xs, y


def preprocess_keras2(dict, fcc, seq_len, normalize=True, shuffle=True, seqs=False):
    '''
    Scales the torque column with a minmax scaler
    Add 2 new rows based on crank angle, removes crank angle column
    Creates sequences of length SEQ_LEN
    :param df: The dataframe
    :return: x: the sequenced data
             y: the resulting data
    '''
    # print("Preprocessing")
    df = pandas.DataFrame(dict)

    min_max_scaler = preprocessing.MinMaxScaler()
    for col in df.columns:
        if normalize and col != df_crank_angle_rad:
            vals_scaled = min_max_scaler.fit_transform(df[[col]])
            df_new = pandas.DataFrame(vals_scaled)
            df[col] = df_new
            # df[col] = preprocessing.scale(df[col].values) todo check if this is better
        if col == df_crank_angle_rad:
            cosvals1 = []
            sinvals1 = []
            for val in df[col].values:
                cosvals1.append(cos(val))
                sinvals1.append(sin(val))
            df['cos_angle'] = pandas.Series(cosvals1)
            df['sin_angle'] = pandas.Series(sinvals1)

    df = df.drop(df_crank_angle_rad, 1)

    sequences = []
    prev_data = deque(maxlen=seq_len)
    j = -1
    for i in df.values:
        prev_data.append([n for n in i])
        if len(prev_data) == seq_len and j != 50:
            sequences.append([np.array(prev_data), fcc[j]])

    if shuffle:
        random.shuffle(sequences)
    x = []
    y = []
    for seq, target in sequences:
        x.append(seq)
        y.append(target)
    # print("Preprocessing Finished")
    xs = np.array(x)
    if not seqs:
        nsamples, nx, ny = xs.shape
        return xs.reshape((nsamples, nx * ny)), y
    return xs, y


def preprocess_keras(dict, fcc, seq_len, normalize=True, shuffle=True, seqs=False):
    '''
    Scales the torque column with a minmax scaler
    Add 2 new rows based on crank angle, removes crank angle column
    Creates sequences of length SEQ_LEN
    :param df: The dataframe
    :return: x: the sequenced data
             y: the resulting data
    '''
    # print("Preprocessing")
    df = pandas.DataFrame(dict)
    speed_scaler = preprocessing.MinMaxScaler((0, 45))
    torque_scaler = preprocessing.MinMaxScaler((0, 60))
    slope_scaler = preprocessing.MinMaxScaler((0, 0.1))
    for col in df.columns:
        if normalize and col != df_crank_angle_rad:
            if col == df_torque:
                vals_scaled = torque_scaler.fit_transform(df[[col]])
            elif col == df_velocity:
                vals_scaled = speed_scaler.fit_transform(df[[col]])
            elif col == df_slope:
                vals_scaled = slope_scaler.fit_transform(df[[col]])
            else:
                vals_scaled = []
                raise ValueError("This shouldn't happen")
            df_new = pandas.DataFrame(vals_scaled)
            df[col] = df_new
        elif col == df_crank_angle_rad:
            cosvals1 = []
            sinvals1 = []
            for val in df[col].values:
                cosvals1.append(cos(val))
                sinvals1.append(sin(val))
            df['cos_angle'] = pandas.Series(cosvals1)
            df['sin_angle'] = pandas.Series(sinvals1)

    df = df.drop(df_crank_angle_rad, 1)
    print(df.head())
    sequences = []
    prev_data = deque(maxlen=seq_len * 5)
    j = 0
    for i in range(0, len(df) - 1):
        for cols in df.iloc[i]:
            prev_data.append(cols)
        if len(prev_data) == seq_len * 5:
            sequences.append([np.array(prev_data), fcc[j]])
            j += 1

    if shuffle:
        random.shuffle(sequences)
    X = []
    y = []
    for seq, target in sequences:
        X.append(seq)
        y.append(target)
    # print("Preprocessing Finished")
    xs = np.array(X)
    if not seqs:
        nsamples, nx, ny = xs.shape
        return xs.reshape((nsamples, nx * ny)), y
    return xs, y


def preprocess_dict(dict, seq_len, normalize=True, shuffle=True, classification=False, seqs=False):
    '''
    Scales the torque column with a minmax scaler
    Add 2 new rows based on crank angle, removes crank angle column
    Creates sequences of length SEQ_LEN
    :param df: The dataframe
    :return: x: the sequenced data
             y: the resulting data
    '''
    df = pandas.DataFrame(dict)
    return preprocess(df, seq_len, normalize=normalize, shuffle=shuffle, classification=classification, seqs=seqs)
