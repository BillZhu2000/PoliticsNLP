"""
Code for scraping Reddit data using PushShift.io instead of normal praw (Python Reddit API wrapper) due to size
constraints imposed by Reddit after they moved off of cloudsearch
"""
import datetime as dt
import re
import time

import pandas as pd
import requests
from nltk.stem import WordNetLemmatizer


# Define fuction to pull subreddit submissions and pull fields from 'subfield'
def query_pushshift(subreddit, kind='submission', skip=30, times=10, subfield=None, comfields=None):
    """
    Query PushShift API to get required information
    :param subreddit:
    :param kind:
    :param skip:
    :param times:
    :param subfield:
    :param comfields:
    :return:
    """
    # Create stem of PushShift API URL + kind and subreddit name
    if comfields is None:
        comfields = ['body', 'score', 'created_utc']
    if subfield is None:
        subfield = ['title', 'selftext', 'subreddit', 'created_utc', 'author', 'num_comments', 'score',
                    'is_self']
    stem = "https://api.pushshift.io/reddit/search/{}/?subreddit={}&size=500".format(kind, subreddit)
    mylist = []

    # Create for loop from 1 to times (specified in )
    for x in range(1, times):
        # Pull json from URL every X days
        URL = "{}&after={}d".format(stem, skip * x)
        print(URL)
        response = requests.get(URL)

        # Check API status
        assert response.status_code == 200
        mine = response.json()['data']

        # Create dataframe from json data dictionary
        df = pd.DataFrame.from_dict(mine)
        mylist.append(df)

        # Set sleep timer
        time.sleep(1)

    # Compile all posts into full list
    full = pd.concat(mylist, sort=False)
    print(list(full))

    # Pull submission subfields and drop duplicates
    if kind == "submission":
        full = full[subfield]
        full = full.drop_duplicates()
        full = full.loc[full['is_self'] != True]

    # Transform UTC into date
    def get_date(created):
        return dt.date.fromtimestamp(created)

    _timestamp = full["created_utc"].apply(get_date)
    full['timestamp'] = _timestamp
    print(full.shape)
    full.to_csv('rsc/{}.csv'.format(subreddit))

    return full


def clean_text(raw):
    # Intantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Remove non-letter characters
    letters_only = re.sub('[^a-zA-z]', ' ', raw)

    # Convert string to lower case
    lower_only = letters_only.lower().split()

    # Lemmatize words in lower_only string
    lemm_words = [lemmatizer.lemmatize(i) for i in lower_only]

    # Return cleaned words if not in stopwords
    words = [w for w in lemm_words]

    # Return clean words and join with space
    return " ".join(words)
