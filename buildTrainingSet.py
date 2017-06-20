#!/usr/bin/env python

import sqlite3
import numpy as np



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
            
            
# Open the SQL database
sqlite_file = 'tweet_db.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()


x = c.execute("SELECT userId, desc FROM user_table WHERE screenName != ' ' ")

userIds = []
catVects = []

for rec in x:
    indicatorVector = [0]*len(categories)
    for cat in range(len(categories)):
        if any(word in rec[1].lower()  for word in keywords[categories[cat]]):
            indicatorVector[cat] = 1
    userIds.append(rec[0])
    catVects.append(indicatorVector)