#!/usr/bin/env python

import re
import sqlite3
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer

# This is my first version of a classifier, inspired by the scikit-learn tutorial:
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#
# Other references:
# https://stackoverflow.com/questions/35612918/create-pandas-dataframe-iteratively
# https://chrisalbon.com/python/pandas_indexing_selecting.html
# http://www.nltk.org/book/ch05.html
# http://www.nltk.org/api/nltk.tokenize.html
# https://stackoverflow.com/questions/35861482/nltk-lookup-error

df = pd.read_pickle('categorizedData.dat')

# Read in categories and keywords
keywords = {}
categories = []
with open('topLevelCategories.txt', 'r') as fin:
    curCat = ""
    for line in fin:
        if not line.startswith('-'):
            curCat = line.rstrip()
            keywords[curCat] = []
            categories.append(curCat)
        elif line.startswith('-'):
            keywords[curCat].append(line[1:].rstrip())


# For each user, get and concatenate tweets

# Open the SQL database
sqlite_file = 'tweet_db.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Database schema
t1 = 'user_table'
t1f1 = 'userId'
t1f2 = 'screenName'
t1f3 = 'English'
t1f4 = 'desc'

t2 = 'tweet_table'
t2f1 = 'tweetId'
t2f2 = 'userId'
t2f3 = 'tweetText'
t2f4 = 'English'

t3 = 'hashtag_table'
t3f1 = 'tweetId'
t3f2 = 'hashtag'

try:
    print('Starting Index Creation')
    c.execute("CREATE INDEX userId_index ON tweet_table (userId)") # This speeds things up immensely
    print('Finished Index Creation')
except sqlite3.OperationalError:
    print('Index already generated')

all_tweets = {}
indexes = []
my_df = []

tknzr =TweetTokenizer()

count = 0
for index, row in df.iterrows():
    count += 1
    if count % 10 == 0:
        print(count)
    tweet_str = ''
    x = c.execute("SELECT tweetText FROM tweet_table WHERE userId == {0}".format(index))
    for i in x:
        tweet_str += i[0] + ' '
        
    # Remove urls
    # Tokenize with special twitter tokenizer
    # Use nltk to Part Of Speeech Tag, and extract all nouns and verbs
    # I'm doing all this here to save memory in what is ultimately stored
    tweet_strs = ' '.join([i[0] for i in nltk.pos_tag(tknzr.tokenize( re.sub(r'http\S+', '', tweet_str).lower())) if i[1]=='NN' or i[1].startswith('V')])
    cats = df.loc[index]
    for ct in range(len(categories)):
        if cats[ct] == 1:
            my_df.append([index+'_'+str(ct), ct, tweet_str])