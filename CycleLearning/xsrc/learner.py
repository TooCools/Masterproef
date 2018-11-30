import tensorflow as tf
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle


def learn_lstm(name, train_x, train_y, val_x, val_y, batch_size=64, epoch=10, save=False):
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    model = Sequential()
    # model.add(LSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
    model.add(CuDNNLSTM(64, input_shape=train_x.shape[1:]))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  # normalizes activation outputs, same reason you want to normalize your input data.

    # model.add(LSTM(128, return_sequences=True))
    # model.add(CuDNNLSTM(128, return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())

    # model.add(LSTM(128))
    # model.add(CuDNNLSTM(128))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Dense(32))
    # model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    # model.add(Dense(1, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['mean_squared_error']
    )
    tensorboard = TensorBoard(log_dir="logs/{}".format(name))

    filepath = "RNN_Final-{epoch:02d}-{mean_squared_error:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint(
        "models/nn/{}.model".format(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True,
                                    mode='min'))  # saves only the best ones
    history = model.fit(
        train_x, train_y,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(val_x, val_y),
        callbacks=[tensorboard, checkpoint],
    )

    score = model.evaluate(val_x, val_y, verbose=0)
    print('Test loss mse:', score[0])
    # Save model
    if save:
        model.save("models/{}".format(name))
    return model


def learn_tree(name, train_x, train_y, depth=10, save=False, classification=False):
    if classification:
        tree = DecisionTreeClassifier(max_depth=depth)
    else:
        tree = DecisionTreeRegressor(max_depth=depth)
    nsamples, nx, ny = train_x.shape
    d2_train_dataset = train_x.reshape((nsamples, nx * ny))
    tree.fit(d2_train_dataset, train_y)
    if save:
        pickle.dump(tree, open("models/trees/{}.sav".format(name), "wb"))
    return tree


def learn_randomforest(name, train_x, train_y, estimators=10, depth=10, save=False, classification=False):
    if classification:
        forest = RandomForestClassifier(max_depth=depth, n_estimators=estimators)
    else:
        forest = RandomForestRegressor(max_depth=depth, n_estimators=estimators)
    nsamples, nx, ny = train_x.shape
    d2_train_dataset = train_x.reshape((nsamples, nx * ny))
    forest.fit(d2_train_dataset, train_y)
    if save:
        pickle.dump(forest, open("models/forests/{}.sav".format(name), "wb"))
    return forest


def learn_PA(name, train_x, train_y, max_iter=100, save=False, classification=False):
    if classification:
        model = PassiveAggressiveClassifier(max_iter=max_iter, warm_start=True,shuffle=True)
    else:
        model = PassiveAggressiveRegressor(max_iter=max_iter, warm_start=True,shuffle=True)
    model.fit(train_x, train_y)
    if save:
        pickle.dump(model, open("models/pa/{}.sav".format(name), "wb"))
    return model


def defaultPA():
    return PassiveAggressiveRegressor(max_iter=100, warm_start=True)
