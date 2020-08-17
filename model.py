"""
NLP and ML semantic modeling of data using Tensorflow NLP library utilities and nltk
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import prep_df


# Suppress TF warnings
tf.get_logger().setLevel('INFO')

# Necessary code to allow GPU to run
physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # tf.config.set_soft_device_placement(enabled=False)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except AssertionError:
    pass


def load_data(file_name):
    """
    Load data and perform preprocessing

    :param file_name:
    :return:
    """
    df = pd.read_csv('rsc/' + file_name, index_col=0)
    df = prep_df(df)
    df['subreddit'] = file_name.split('.')[0]
    return df


def plot_graphs(history, string):
    """
    print loss and accuracy

    :param history:
    :param string:
    :return:
    """
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def recall_m(y_true, y_pred):
    """
    Calculate recall for batch

    :param y_true: Actual y value
    :param y_pred: Calculated y value
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Calculate precision for batch

    :param y_true:
    :param y_pred:
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    Calculate F1 measure for batch

    :param y_true:
    :param y_pred:
    :return:
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def main():
    """
    Build and train model

    :return:
    """
    # intiailize params and defaults
    vocab_size = 100000
    embedding_dim = 64
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # Train against republicans and democrats
    democrat = load_data('Democrats.csv')
    republican = load_data('politics.csv')
    democrat['label'] = 0
    republican['label'] = 1
    comb_df = pd.DataFrame()
    comb_df = comb_df.append(democrat, ignore_index=True)
    comb_df = comb_df.append(republican, ignore_index=True)

    # prep train and test sets
    X = np.array(comb_df['title'])
    y = np.array(comb_df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(X_train)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    testing_sequences = tokenizer.texts_to_sequences(X_test)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

    # Build model, currently multiple bidirectional LSTMs
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', dropout=0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.5)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.5)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1024, activation='relu', dropout=0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Convolution model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(len(word_index), embedding_dim, input_length=max_length),
    #     tf.keras.layers.Conv1D(256, 5, activation='relu'),
    #     tf.keras.layers.MaxPooling1D(),
    #     tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Conv1D(128, 5, activation='relu'),
    #     tf.keras.layers.MaxPooling1D(),
    #     tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Conv1D(64, 5, activation='relu'),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(24, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])

    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy', precision_m, recall_m, f1_m],
                  run_eagerly=True)

    NUM_EPOCHS = 10
    history = model.fit(padded, y_train, epochs=NUM_EPOCHS, validation_data=(testing_padded, y_test), batch_size=50)
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    plot_graphs(history, 'precision_m')
    plot_graphs(history, 'recall_m')
    plot_graphs(history, 'f1_m')


if __name__ == '__main__':
    main()
