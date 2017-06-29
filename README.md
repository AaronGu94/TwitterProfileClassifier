# TwitterProfileClassifier

A project that builds classifiers identifying what topics a Twitter user tweets on. The classification is inspired by the work found at http://varianceexplained.org/programming/tagger-news/ .

Requires that the user be registered with twitter and have a CONSUMER_KEY and CONSUMER_SECRET. This are placed, each on their own line, in a file with the name `KeyAndSecret.txt`

The code is a little messy, but is intended to be run in the following order:

### getTwitterIDs.py
Calls the Twitter APIs to get the IDs of followers and followees (friends) of people I (twitter username: nbryans) follow.


### getTweetsAndProfileDetails.py
For the user IDs found in the previous module, this calls the Twitter APIs to get profile information (i.e. description and username) along with the most recent 100 tweets.

This data is saved in a SQL (sqlite3) database on a periodic basis as it is downloading. This module is very slow as tweets for only 1500 people can be obtained every 15 minutes (see Twitter API Limits).

### (Optional) summarizeData.py
Summarizes the data found in the SQL database.

### buildTrainingSet.py
Classifies each user in the database with tweets by looking at their description and comparing the words against those in the file `topLevelCategories.txt`

### LDA_RanForest_Classifier.py
Builds a classifier using `gensim` Latent Dirichlet Allocation and Random Forests. Tests the accuracy of this classifier.

### TFIDF_Classifiers.py
Builds a classifier (inspired by the scikit-learn tutorial: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) using Term Frequency times Inverse Document Frequency (TFIDF) and either Multinomial Naive Bayes or Random Forests. Tests the accuracy of this classifier.

### LearningFromTwitterData.pdf
The presentation I created to summarize this work and present it at the June PrairieML meeting.