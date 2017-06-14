#!/usr/bin/env python

import json
from application_only_auth import Client
import time
import os.path
import pickle
import sqlite3

# Get the key and the secret. Saved in a seperate file for security reasons.
# You can request your own via: https://dev.twitter.com/apps/new
with open('KeyAndSecret.txt') as fin:
    CONSUMER_KEY = fin.readline().rstrip()
    CONSUMER_SECRET = fin.readline().rstrip()
    
client = Client(CONSUMER_KEY, CONSUMER_SECRET)


# See how many Unique IDs we retrieved in the previous step
numBlocks = 10
levelTwoIDsRaw = []

for i in range(numBlocks):
    with open('block_{0}.dat'.format(str(i)), 'r') as fin:
        for line in fin:
            levelTwoIDsRaw.append(line.rstrip().replace('[',' ').replace(']',' ').split(','))

levelTwoIDs = [j.strip() for i in levelTwoIDsRaw for j in i]
uLevelTwoIDs = list(set(levelTwoIDs))
uLevelTwoIDs.sort()
uLevelTwoIDs = uLevelTwoIDs[1:] # Clear the opening empty value


# Create a SQLite Database, based partially on the tutorial here: http://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html#connecting-to-an-sqlite-database
sqlite_file = 'tweet_db.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()


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
'''
c.execute('CREATE TABLE {tn} ({nf} {ft} PRIMARY KEY)'.format(tn=t1, nf=t1f1, ft='INTEGER'))
c.execute('CREATE TABLE {tn} ({nf} {ft} PRIMARY KEY)'.format(tn=t2, nf=t2f1, ft='INTEGER'))

c.execute('CREATE TABLE {tn} ({nf} {ft}, {cn} {ct},  PRIMARY KEY({nf}, {cn}))'.format(tn=t3, nf=t3f1, ft='INTEGER', cn=t3f2, ct='TEXT'))

c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}".format(tn=t1, cn=t1f2, ct='TEXT'))
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}".format(tn=t1, cn=t1f3, ct='INTEGER'))
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}".format(tn=t1, cn=t1f4, ct='TEXT'))

c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct} DEFAULT '{df}' NOT NULL".format(tn=t2, cn=t2f2, ct='INTEGER', df=0))
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}".format(tn=t2, cn=t2f3, ct='TEXT'))
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}".format(tn=t2, cn=t2f4, ct='INTEGER'))

conn.commit()

for i in range(len(uLevelTwoIDs)):
    try:
        c.execute("INSERT INTO {tn} ({idf}, {cn2}, {cn3}, {cn4}) VALUES ({id},  {defName}, 1, {defDesc})".format(tn=t1, idf=t1f1, cn2=t1f2, cn3=t1f3, cn4=t1f4, id=long(uLevelTwoIDs[i]), defName="' '", defDesc="' '"))
    except sqlite3.IntegrityError:
        print('ERROR: ID already exists in PRIMARY KEY column {0} {1}'.format(t1f1, str(levelTwoIDs[i])))
        
conn.commit()
'''

# Query User's Timelines
numCurRec = 1490
secondsBetweenQuery = 15*60.0
blockNum = 0
start=600



currentRecords = []

for i in range(start, len(uLevelTwoIDs)):
    if i % 100 == 0:
        print("Starting record {0}".format(str(i)))
    if i % 700 == 0:
        time.sleep(100.0)
    try:
        tweetData = client.request('https://api.twitter.com/1.1/statuses/user_timeline.json?user_id={id}&count={cnt}'.format(id=str(uLevelTwoIDs[i]), cnt=100))
        
        currentRecords.append(tweetData)
    except:
        print('Error searching id: {id}'.format(id=str(uLevelTwoIDs[i])))

    if len(currentRecords) % 100 == 0:
        for rec in currentRecords:
            # Get User Data
            if len(rec) > 0:
                user_id = rec[0]['user']['id']
                user_screen_name = rec[0]['user']['screen_name'].encode('utf-8').replace("'", "").replace('"', '')
                user_description = rec[0]['user']['description'].encode('utf-8').replace("'", "").replace('"', '')
                user_lang =  1 if rec[0]['user']['lang'] == 'en' else 0
                
                c.execute("UPDATE {tn} SET {cn2}=('{v2}'), {cn3}=({v3}), {cn4}=('{v4}') WHERE {idc}=({id})".format(tn=t1, cn2=t1f2, cn3=t1f3, cn4=t1f4, v2=user_screen_name, v3=user_lang, v4=user_description,idc=t1f1, id=int(user_id)))
                
                for tweet in rec:
                    tweet_text = tweet['text'].encode('utf-8').replace("'", "").replace('"', '')
                    tweet_id = tweet['id']
                    tweet_lang = 1 if tweet['lang'] == 'en' else 0
                    
                    for h in tweet['entities']['hashtags']:
                        try:
                            c.execute("INSERT INTO {tn} ({c1}, {c2}) VALUES ({v1}, '{v2}')".format(tn=t3, c1=t3f1, c2=t3f2, v1=tweet_id, v2=h['text'].encode('utf-8').replace("'", "").replace('"', '')))
                        except sqlite3.IntegrityError:
                            pass # Sometimes tweet contains same hashtag twice
                        
                    c.execute("INSERT INTO {tn}  ({c1}, {c2}, {c3}, {c4})  VALUES ({v1}, {v2}, '{v3}', {v4})".format(tn=t2, c1=t2f1, c2=t2f2, c3=t2f3, c4=t2f4, v1=tweet_id, v2=user_id, v3=tweet_text,v4=tweet_lang))
                conn.commit()
        currentRecords = []


    # print('Waiting {0} seconds.'.format(str(secondsBetweenQuery - (time.time() - starttime))))
    # time.sleep(secondsBetweenQuery - (time.time() - starttime) + 60.0) # The extra 60 seconds is a buffer

conn.close()
