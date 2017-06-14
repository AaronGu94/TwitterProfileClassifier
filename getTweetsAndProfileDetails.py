#!/usr/bin/env python

# See how many Unique IDs we retrieved in the previous step
numBlocks = 10
levelTwoIDsRaw = []

for i in range(numBlocks):
    with open('block_{0}.dat'.format(str(i)), 'r') as fin:
        for line in fin:
            levelTwoIDsRaw.append(line.rstrip().replace('[',' ').replace(']',' ').split(','))

levelTwoIDs = [j.strip() for i in levelTwoIDsRaw for j in i]
