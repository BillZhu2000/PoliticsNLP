"""
Preprocessing utilities
"""

import os
import re
import string

import pandas as pd
from nltk.stem import WordNetLemmatizer


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
    text = re.sub(r'\w*\d\w*', '', text)
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
    df['selftext'] = pd.Series([re.sub(r'(http)\S*', ' ', elem) for elem in df['selftext']])
    df['selftext'] = pd.Series([re.sub('\\n', ' ', elem) for elem in df['selftext']])
    df['selftext'] = pd.Series([re.sub(r'\[(link)\]', ' ', elem) for elem in df['selftext']])
    df['selftext'] = pd.Series([re.sub('(&gt)', ' ', elem) for elem in df['selftext']])

    # Clean title & selftext
    df['title'] = pd.Series([clean_text(str(elem)) for elem in df['title']])
    df['selftext'] = pd.Series([clean_text(str(elem)) for elem in df['selftext']])

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


if __name__ == '__main__':
    main()
