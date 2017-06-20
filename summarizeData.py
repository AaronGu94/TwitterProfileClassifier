#!/usr/bin/env python

import sqlite3
import numpy as np

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

# Find number of users with tweet data (they will have a screen name filled in)
x = c.execute("SELECT * FROM user_table WHERE screenName !=  ' ' ")
print('There are {0} users.'.format(len(list(x))))

# Find the average number of words per description
x = c.execute("SELECT desc FROM user_table WHERE screenName != ' ' ")
numWords = [len(i[0].split()) for i in list(x)]
print('There are, on average, {0} words per user description.'.format(np.mean(numWords)))

# Find the average number of tweets per user
x = c.execute("SELECT userId FROM user_table WHERE screenName != ' ' ")
IDs = [i[0] for i in list(x)]

print("Starting Index Creation")
c.execute("CREATE INDEX userId_index ON tweet_table (userId)") # This speeds things up immensely
print("Finished Index Creation")

numTweets = []
for i in IDs:
    x = c.execute("SELECT * FROM tweet_table WHERE userId == {0}".format(i))
    numTweets.append(len(list(x)))
print("There are, on average, {0} tweets per user.".format(np.mean(numTweets)))