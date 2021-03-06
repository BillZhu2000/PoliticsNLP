# PoliticsNLP
NLP classification on politics forums

This repository is for NLP classification of Reddit submissions and Twitter posts, among other political forums. The goal
is to classify submissions and see where they fall on the political spectrum. Both supervised and unsupervised methods
will be used to see if there are any unusual patterns.

Currently, ```model.py``` is configured to train on reddit subreddit data and classify between two different subreddits 
based on title alone. Currently, differentiating between Republicans and Democrats gets around 75% accuracy, while between
Socialism and Communism, two very similar subreddits, is at ~65 to 68, which is still substantially better than 50% baseline
(i.e. flipping a coin). Between Democrats and Politics, the similarity was also 75%, meaning there are some differences
between the two subreddits (r/politics is rather liberal).