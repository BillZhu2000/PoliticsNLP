"""
Preprocessing utilities
"""

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
import os
import re
import string


def drop_punc(text):
    """
    Drop punctuation from text
    :param text:
    :return:
    """
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return text


def remove_numbers(text):
    """
    Remove digits
    :param text:
    :return:
    """
    text = re.sub('\w*\d\w*', '', text)
    return text


def deEmojify(inputString):
    """
    Drop emojis
    :param inputString:
    :return:
    """
    return inputString.encode('ascii', 'ignore').decode('ascii')


def label_load_data(file_dir, df):
    """
    Label the data according to the given
    :return:
    """
    counter = 0
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            df1 = pd.read_csv(os.path.join(dirpath, file))
            df1['subreddit'] = file.split('.')[0]
            df1['label'] = counter
            counter += 1
            df = df.append(df1, ignore_index=True)
    df.drop(columns='Unnamed: 0', inplace=True)
    print(df.head())
    return df


def prep_df(df):
    """
    Clean dataframe procedures
    :param df:
    :return:
    """
    # Drop rows with empty title
    df = df[df['title'] != '']

    # Fill '[removed]' with blank
    df.loc[df['selftext'] == '[removed]', 'selftext'] = ' '
    df.loc[df['selftext'] == '[deleted]', 'selftext'] = ' '
    df.loc[df['selftext'] == '\[removed\]', 'selftext'] = ' '
    df.loc[df['selftext'] == '&amp;#x200B;', 'selftext'] = ' '

    # Fill nan with blank
    df['selftext'].fillna('', inplace=True)

    # Drop rows that have been deleted by moderators
    df = df[~df['selftext'].str.contains("Feel free to discuss")]
    df.reset_index(drop=True, inplace=True)

    # Remove hyperlinks and \n line breaks from selftext
    for i in range(0, len(df)):
        df.loc[i, 'selftext'] = re.sub('(http)\S*', ' ', str(df.loc[i, 'selftext']))
        df.loc[i, 'selftext'] = re.sub('\\n', ' ', str(df.loc[i, 'selftext']))
        df.loc[i, 'selftext'] = re.sub('\[(link)\]', ' ', str(df.loc[i, 'selftext']))
        df.loc[i, 'selftext'] = re.sub('(&gt)', ' ', str(df.loc[i, 'selftext']))

    # Clean title & selftext
    for i in range(0, len(df)):
        df.loc[i, 'title'] = clean_text(str(df.loc[i, 'title']))
        df.loc[i, 'selftext'] = clean_text(str(df.loc[i, 'selftext']))

    # Create one combined_text column to clean
    df['combined_text'] = df['title'] + " " + df['selftext']

    # Drop rows with less than 10 words
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
    df.reset_index(drop=True, inplace=True)
    df = df[df['word_count'] > 5]
    print(df.head())
    return df


def clean_text(raw):
    """
    Clean text by lemmatizing, lowercasing, and removing stop words
    :param raw:
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    letters_only = re.sub('[^a-zA-z]', ' ', raw)
    lower_only = letters_only.lower().split()
    lemm_words = [lemmatizer.lemmatize(i) for i in lower_only]
    cleaned_words = [w for w in lemm_words]

    # Drop punc, deEmoji, etc.
    reconstruct = " ".join(cleaned_words)
    return drop_punc(deEmojify(reconstruct))


def main():
    file_dir = 'rsc'
    df = pd.DataFrame()

    # Perform preprocessing
    df = label_load_data(file_dir, df)
    df = prep_df(df)
    df.to_csv('')
