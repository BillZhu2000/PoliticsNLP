"""
NLP and ML semantic modeling of data using Tensorflow NLP library utilities and nltk
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import prep_df

# Necessary code to allow GPU to run
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


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
    plt.plot(history.history('val_' + string))
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def main():
    # intiailize params and defaults
    vocab_size = 10000
    embedding_dim = 64
    max_length = 120
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # Train against republicans and democrats
    democrat = load_data('Democrats.csv')
    republican = load_data('Republican.csv')
    democrat['label'] = 0
    republican['label'] = 1
    comb_df = pd.DataFrame()
    comb_df = comb_df.append(democrat, ignore_index=True)
    comb_df = comb_df.append(republican, ignore_index=True)

    # prep train and test sets
    X = np.array(comb_df['title'])
    y = np.array(comb_df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
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
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    NUM_EPOCHS = 10
    history = model.fit(padded, y_train, epochs=NUM_EPOCHS, validation_data=(testing_padded, y_test))
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')


if __name__ == '__main__':
    main()
