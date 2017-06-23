#!/usr/bin/env python

import re
import os
import nltk
import sqlite3
import numpy as np
import pandas as pd
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

if os.path.isfile('tokenizedNounsAndVerbs.dat'):
    print("Pickled File Exists")
    df2 = pd.read_pickle('tokenizedNounsAndVerbs.dat')
else:
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
        tweet_strToken = ' '.join([i[0] for i in nltk.pos_tag(tknzr.tokenize( re.sub(r'http\S+', '', tweet_str).lower())) if i[1]=='NN' or i[1].startswith('V')])
        cats = df.loc[index]
        for ct in range(len(categories)):
            if cats[ct] == 1:
                my_df.append([index+'_'+str(ct), ct, tweet_strToken])
        
    df2 = pd.DataFrame(my_df)
    df2.to_pickle('tokenizedNounsAndVerbs.dat')

# Partitioning DataFrame into Training and Testing datasets
mask = np.random.rand(len(df2)) < 0.8 # Test set made from ~ 80% of the data
trainData = df2[mask]
testData = df2[~mask]


# Getting Counts of test data
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
training_counts = count_vect.fit_transform(trainData[2])
# print(training_counts.shape) # Print the shape of this sparse matrix
# print(count_vect.vocabulary_.get(u'science') # Get the index (from the count_vect dictionary) of the word 'science'


# Moving from occurrences to frequencies
from sklearn.feature_extraction.text import TfidfTransformer
# tf_transformer = TfidfTransformer(use_idf=False).fit(training_counts)
# training_tf = tf_transformer.transform(training_counts)
tfidf_transformer = TfidfTransformer()
training_tfidf = tfidf_transformer.fit_transform(training_counts)
# print(training_tf.shape) # The shape of the Term Frequencies sparse matrix, should be the same as the counts
# print(training_tfidf.shape) # The shape of the Term Frequencies sparse matrix, should be the same as the counts


# Training a classifier
from sklearn.naive_bayes import MultinomialNB # Multinomial is most suitable for word counts
clf = MultinomialNB().fit(training_tfidf, trainData[1])

#predicting output of test set using classifier clf
testing_counts = count_vect.transform(testData[2])
testing_tfidf = tfidf_transformer.transform(testing_counts)
predicted = clf.predict(testing_tfidf)


# Check accuracy
print(" {0}/{1} were found to have been predicted correctly".format(np.mean(testData[1] == predicted), len(testData[1])))
# np.bincount(predicted) # Shows how many of each class predicted