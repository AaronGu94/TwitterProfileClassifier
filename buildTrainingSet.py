#!/usr/bin/env python

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

catVects = {}

for rec in x:
    indicatorVector = [0]*len(categories)
    flag = 0
    for cat in range(len(categories)):
        if any(word in rec[1].lower()  for word in keywords[categories[cat]]):
            indicatorVector[cat] = sum([word in rec[1].lower()  for word in keywords[categories[cat]]])
            flag = 1
    if flag == 1: # Only add entries matching some category to the dictionary
        catVects[str(rec[0])] = indicatorVector
    
numWithData = 0
for key in catVects.keys():
    if 1 in catVects[key]:
        numWithData += 1
print("There are {0} users that are classified to some category.".format(numWithData))

# Construct padas DataFrame from the data
df = pd.DataFrame.from_dict(catVects, orient='index') # The orient parameter uses dict keys as row labels
df.columns = categories # Rename the columns

df.to_pickle('categorizedData.dat')

# Plot categories vs. number of users
# df.sum() # See how many users we have for each category
# np.mean(df.sum()) # get average number of users per category
plt.bar(range(len(categories)), list(df.sum()), color='blue', alpha=0.5)
plt.xticks(range(len(categories)), categories, rotation='vertical')
plt.tight_layout()
# plt.show()


# df.sum() Results
"""
Advice                  315
Animals                 965
Art and Design         1695
Books                   674
Business               2032
Celebrity                30
Comics                  219
DIY and Crafts          307
Education              1440
Electronics            1602
Fashion                 312
Food & Drink           1692
Funny                   191
Gaming                  871
Health                  458
Jobs                    137
Military                728
Movies                  236
Music                   902
News                    643
Philosophy              297
Photography             452
Politics                991
Religion                303
Science                1714
Security                173
Sports                 1544
Technology             2077
Television              926
Travel and Outdoors     679
Vehicles                811
"""