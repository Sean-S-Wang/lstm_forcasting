from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def multi_step_plot(history, true_future, prediction, STEP=6):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def univariate_main():
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    # zip_path = tf.keras.utils.get_file(
    #     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    #     fname='jena_climate_2009_2016.csv.zip',
    #     extract=True)
    # csv_path, _ = os.path.splitext(zip_path)

    csv_path = Path(r"C:\Users\Sean\Documents\Software_Projects\forcasting\Forcasting\master.csv")
    df = pd.read_csv(str(csv_path.absolute()))
    # Glance at data
    df.head()

    # Train set is 75% of data
    TRAIN_SPLIT = int(np.floor(df.shape[0] * 0.75))  # 300000
    # Set seed to reproduce
    tf.random.set_seed(13)

    # Part 1: Forecast a univariate time series

    # Get univariate data of interest
    uni_data = df['accountserviceprdr2_callsperminute']
    uni_data.index = df['timestamp']
    # Drop nan values
    uni_data = uni_data.interpolate()
    # Plotting and normalizing
    uni_data.head()
    uni_data.plot(subplots=True)

    uni_data_values = uni_data.values

    uni_train_mean = uni_data_values[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data_values[:TRAIN_SPLIT].std()
    uni_data_values = (uni_data_values-uni_train_mean)/uni_train_std
    # Set windows for past and future
    univariate_past_history = 720
    univariate_future_target = 30

    x_train_uni, y_train_uni = univariate_data(uni_data_values, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data_values, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    print('Single window of past history : {}'.format(x_train_uni[0].shape))
    print('\n Target temperature to predict : {}'.format(y_train_uni[0].shape))

    show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

    # RNN implementation
    simple_lstm_model, val_univariate = rnn_implementation(x_train_uni, x_val_uni, y_train_uni, y_val_uni)

    # Predict
    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                          simple_lstm_model.predict(x)[0:5]], 0, 'Simple LSTM model')
        plot.show()


def rnn_implementation(x_train_uni, x_val_uni, y_train_uni, y_val_uni):
    # RNN Implementation
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    simple_lstm_model.compile(optimizer='adam', loss='mae')
    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)
    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)
    return simple_lstm_model, val_univariate


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def multi_step_model_implementation(train_data_multi, val_data_multi, x_train_multi):
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(72))
    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=50)
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

    return multi_step_model


def main():
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False

    csv_path = Path(r"C:\Users\Sean\Documents\Software_Projects\forcasting\Forcasting\master.csv")
    df = pd.read_csv(str(csv_path.absolute()))
    # Glance at data
    df.head()

    # Train set is 75% of data
    TRAIN_SPLIT = int(np.floor(df.shape[0] * 0.75))  # 300000
    # Set seed to reproduce
    tf.random.set_seed(13)

    # Part 1: Forecast a univariate time series

    # Get univariate data of interest
    features_considered = ['accountserviceprdr3_callsperminute']
    features = df[features_considered]
    features.index = df['timestamp']

    # Drop nan values
    features = features.interpolate()

    features.head()

    features.plot(subplots=True)

    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    past_history = 720
    future_target = 72
    STEP = 6

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)

    print('Single window of past history : {}'.format(x_train_multi[0].shape))
    print('\n Target number of calls to predict : {}'.format(y_train_multi[0].shape))

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]))

    multi_step_model = multi_step_model_implementation(train_data_multi, val_data_multi, x_train_multi)

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


if __name__ == "__main__":
    main()
