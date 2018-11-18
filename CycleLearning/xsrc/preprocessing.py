from sklearn import preprocessing
import pandas
from collections import deque
import numpy as np
from xsrc.params import *
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
    print("Preprocessing")
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in df.columns:
        if normalize and col != df_fcc and col != df_crank_angle_rad:
            vals_scaled = min_max_scaler.fit_transform(df[[col]])
            df_new = pandas.DataFrame(vals_scaled)
            df[col] = df_new
            # df[col] = preprocessing.scale(df[col].values) todo check if this is better
        if col == df_crank_angle_rad:
            df['cos_angle'] = pandas.Series(np.cos(df[col].values))
            df['sin_angle'] = pandas.Series(np.sin(df[col].values))

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
    print("Preprocessing Finished")
    xs = np.array(x)
    if not seqs:
        nsamples, nx, ny = xs.shape
        return xs.reshape((nsamples, nx * ny)),y
    return xs, y
