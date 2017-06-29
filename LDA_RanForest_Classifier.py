#!/usr/bin/env python

import re
import os
import nltk
import sqlite3
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from gensim import corpora, models

# This is classifier is inspired by the work here: http://varianceexplained.org/programming/tagger-news/
# as it uses Latent Dirichlet Allocation as a means of dimensionality reduction to try and improve the size of the problem

def checkAccuracyAllCats(pred, df, testData):
    # A more accurate way to check accuracy
    numCorrect = 0
    count = 0
    for i in testData[0]:
        ansVec = df.loc[i]
        for j in range(len(ansVec)):
            if not ansVec[j] == 0 and pred[count] == j:
                numCorrect += 1
        count += 1
    return float(numCorrect)/len(testData[1])

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
        if sum(df.loc[index] == max(df.loc[index])) == 1:
            for ct in range(len(categories)):
                if cats[ct] == max(df.loc[index]):
                    my_df.append([index, ct, tweet_strToken])
        
    df2 = pd.DataFrame(my_df)
    df2.to_pickle('tokenizedNounsAndVerbs.dat')

# Partitioning DataFrame into Training and Testing datasets
mask = np.random.rand(len(df2)) < 0.8 # Test set made from ~ 80% of the data
trainData = df2[mask]
testData = df2[~mask]

# Code based off of this: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

numTopics = 50

if os.path.isfile('tdweet.lda'):
    ldamodel = models.ldamodel.LdaModel.load('tweet.lda')
else:
    dictionary = corpora.Dictionary([i.split() for i in trainData[2]])
    corpus = [dictionary.doc2bow(text.split()) for text in trainData[2]]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=numTopics, id2word = dictionary, passes = 20)

    ldamodel.save('tweet.lda')

y = ldamodel[[dictionary.doc2bow(text.split() for text in testData[2])]]

# Add new columns to the dataframe
for i in range(numTopics):
    testData[str(i+3)] = 0
    trainData[str(i+3)] = 0
    
dtsTrain = []
for index, row in trainData.iterrows():
    y = ldamodel[dictionary.doc2bow(trainData[2][index].split())]
    new_dict = {}
    for i in range(50):
        new_dict[i] = 0
    for i in y:
        new_dict[i[0]] = float(i[1])
    dtsTrain.append(new_dict)
        

dtsTest = []
for index, row in trainData.iterrows():
    y = ldamodel[dictionary.doc2bow(trainData[2][index].split())]
    new_dict = {}
    for i in range(50):
        new_dict[i] = 0
    for i in y:
        new_dict[i[0]] = float(i[1])
    dtsTrain.append(new_dict)
    
    
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier( n_jobs=2)
clf.fit(pd.DataFrame(dtsTrain), trainData[1])

predicted = clf.predict(pd.DataFrame(dtsTest))
print("Random Forest: {0}% were found to have been predicted correctly".format(100.0*np.mean(testData[1] == predicted)))
print("Random Forest: {0}% were found to have been predicted correctly (for any category)".format(100.0*checkAccuracyAllCats(predicted2, df, testData)))