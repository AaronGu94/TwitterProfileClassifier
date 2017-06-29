#!/usr/bin/env python

import json
from application_only_auth import Client
import time
import os.path
import pickle

'''
see: https://dev.twitter.com/oauth/application-only
Obtain a copy of twitter-application-only-auth here:
https://pypi.python.org/pypi/twitter-application-only-auth/0.3.1 
(credit: Pablo Seminario, Rafael Reimberg and Chris Hawkins)
'''

# Get the key and the secret. Saved in a seperate file for security reasons.
# You can request your own via: https://dev.twitter.com/apps/new
with open('KeyAndSecret.txt') as fin:
    CONSUMER_KEY = fin.readline().rstrip()
    CONSUMER_SECRET = fin.readline().rstrip()
    
client = Client(CONSUMER_KEY, CONSUMER_SECRET)

# status = client.rate_limit_status()
# print status['resources']


# Level 1 - Get a list of my followers
# https://dev.twitter.com/rest/reference/get/friends/ids
MY_USERNAME = 'nbryans'
levelOneFileName = 'levelOne.dat'

if os.path.isfile(levelOneFileName):
    print('Reading pickled data')
    with open(levelOneFileName, 'rb') as fin:
        following = pickle.load(fin)
else:
    followingData = client.request('https://api.twitter.com/1.1/friends/ids.json?cursor=-1&screen_name=' + MY_USERNAME + '&count=5000')
    following = followingData['ids']
    with open(levelOneFileName, 'wb') as fout:
        pickle.dump(following, fout)


# Level 2 - Get a list of those following my followers
# https://dev.twitter.com/rest/reference/get/followers/ids
levelTwoFileName = 'levelTwo.dat'
numRecordAtATime = 14
secondsBetweenQuery = 15*60.0
# blockNum = 0
# time.sleep(secondsBetweenQuery)
# blockNum = 8
# levelTwoFollowers = []

cursors = [-1]*len(following)

if os.path.isfile(levelTwoFileName):
    pass
    # with open(levelTwoFileName, 'rb') as fin:
        # levelTwoFollowers = pickle.load(fin)
else:
    while True:
        levelTwoFollowers = []
        print('Starting Block {0}'.format(str(blockNum)))
        starttime = time.time()
        for i in range(numRecordAtATime*blockNum, min(numRecordAtATime*(blockNum+1), len(following))):
            try:
                followersIDs = client.request('https://api.twitter.com/1.1/followers/ids.json?cursor='+ str(cursors[i]) + '&user_id='+ str(following[i]) + '&count=5000')
                
                friendsIDs = client.request('https://api.twitter.com/1.1/friends/ids.json?cursor='+ str(cursors[i]) + '&user_id='+ str(following[i]) + '&count=5000')
                
                # # There is more data to read - This would add 50 hours for someone with 1 million followers. Not a good approach
                # if not dataIDs['next_cursor'] == 0:
                    # following.append(i)
                    # cursors.append(dataIDs['next_cursor'])
                    
                levelTwoFollowers.append(followersIDs['ids'])
                levelTwoFollowers.append(friendsIDs['ids'])
            except:
                print('Error searching blockNum: {0} corresponding to id: {1} and cursor {2}\n'.format(str(blockNum), str(following[i]), str(cursors[i])))
                pass
        
        with open('block_' + str(blockNum) + '.dat', 'w') as fout:
            fout.write(str(levelTwoFollowers))
            
        blockNum += 1
        
        # Loop exit condition
        if numRecordAtATime*blockNum > len(following):
            break
        
        print('Waiting {0} seconds.'.format(str(secondsBetweenQuery - (time.time() - starttime))))
        time.sleep(secondsBetweenQuery - (time.time() - starttime) + 60.0) # The extra 60 seconds is a buffer
