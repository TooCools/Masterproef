import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras import metrics

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time


def learn_nn(name, train_x, train_y, val_x, val_y, batch_size=64, epoch=10):
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    model = Sequential()
    # model.add(LSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
    model.add(CuDNNLSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  # normalizes activation outputs, same reason you want to normalize your input data.

    # model.add(LSTM(128, return_sequences=True))
    model.add(CuDNNLSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    # model.add(LSTM(128))
    model.add(CuDNNLSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

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
        "models/{}.model".format(filepath, monitor='mean_squared_error', verbose=1, save_best_only=True,
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
    model.save("models/{}".format(name))
    return model


def learn_lstm():
    print("lstm")
