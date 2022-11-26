# Sentiment Analysis Using Gaussian Naive Bayesian Classifier
In this repository we are going to use Gaussian naive bayesian classifer for sentiment analysis. We are going to use twitter airline sentiment data given in the kaggle.


## Preparation of Data

We have Tweets.csv which is a real data set from Kaggle: https://www.kaggle.com/
datasets/crowdflower/twitter-airline-sentiment  
We also have a program tweet prep data.py that:  
• Reads Tweets.csv using the csv library.  
• Discards tweets where the labeler was not at least 50  
• Ignores all stop words and names of airlines  
• Saves out a list of the 2000 most common words that remain.  
• Splits the remaining data into train tweet.csv and test tweet.csv. About 10% of the
tweets will end up in test tweet.csv.  
• Prints out the 32 most common words.  
If you haven’t already, you will need to install nltk and its English stopwords:  
> pip3 install nltk  
> python3  
Type "help", "copyright", "credits" or "license" for more information.  
  
>import nltk  
>nltk.download(’stopwords’)  
  
You do not need to change tweet prep data.py at all. Run it to create vocablist tweet.pkl,
train tweet.csv, and test tweet.csv.  
Here’s what it should look like when you run it:  
> python3 tweet_prep_data.py  
  
Kept 14404 rows, discarded 236 rows  
Most common 32 words are [’flight’, ’get’, ’cancelled’, ’thanks’, ’service’,
’help’, ’time’, ’customer’, ’im’, ’us’, ’hours’, ’flights’, ’hold’, ’amp’,
’plane’, ’thank’, ’cant’, ’still’, ’one’, ’please’, ’need’, ’would’, ’delayed’,
gate’, ’back’, ’flightled’, ’call’, ’dont’, ’bag’, ’hour’, ’got’, ’late’]  
Wrote 2000 words to vocablist_tweet.pkl.  
(Why is ”flightled” on this list? I have no idea. Real data is sometimes weird.)

## Gaussian Classifier and basic statistics

We are given train gn.csv. We will write a program called gnclassifier.py.
The data in train gn.csv looks like this:  
X0,X1,X2,X3,Y  
3.42,43.62,11.68,10.17,PY9  
3.07,88.66,10.01,20.09,PY9  
1.69,16.20,7.81,10.70,TK1  
1.04,35.51,8.37,16.85,RM9  
3.50,59.01,10.24,10.73,PY9  
The last column is the class of that datapoint. That is what you are trying to predict.  
We will compute the mean and standard deviation of each attribute for each class  
We will also figure out the prior for each class.  
We will store the labels, the priors, the means, and the standard deviations in parameters gn.pkl.  
When it runs, it should look like this:  
> python3 gn_train.py  
  
Read 1198 samples with 4 attributes from train_gn.csv  
Priors:  
D6X: 37.3%  
PY9: 35.7%  
RM9: 19.9%  
TK1: 5.0%  
ZZ4: 2.1%  
D6X:  
Means -> X0: 4.0291  X1:33.4690  X2: 6.8480  X3: 7.3198
Stdvs -> X0: 0.5664  X1:16.4635  X2: 4.1119  X3: 5.3221
PY9:  
Means -> X0: 3.0236  X1:62.7718  X2: 7.0495  X3:15.4692
Stdvs -> X0: 0.5868  X1:23.4633  X2: 2.9422  X3: 5.7730
RM9:  
Means -> X0: 1.9467  X1:24.9554  X2: 6.9616  X3:10.4789
Stdvs -> X0: 0.5615  X1: 6.8383  X2: 1.8640  X3: 2.7498
TK1:
Means -> X0: 1.0353  X1:13.8293  X2: 7.0343  X3:10.5025
Stdvs -> X0: 0.6032  X1: 9.0163  X2: 1.1721  X3: 2.5536
ZZ4:
Means -> X0: 4.9596  X1:14.5060  X2: 5.5684  X3: 1.8624
Stdvs -> X0: 0.5395  X1: 4.5257  X2: 6.3474  X3: 5.6984
Wrote parameters to parameters_gn.pkl

## Training 

We will create a second program called gn train.py. It will read parameters gn.pkl. (Be sure
to take the log of the priors before adding them to the log likelihoods!)
Then it will go through each row of test gn.csv and use a Gaussian Naive Bayes approach to
predict the class for that row.
Print the probabilities of each class for the first ten rows of data.
Then you will produce the same sorts of metrics that you did for the tweets.
When it runs, the command line will look like this:

> python3 gn_test.py  
  
Read parameters from parameters_gn.pkl  
Can expect 37.3% accuracy by guessing "D6X" every time.  
Read 302 rows from test_gn.csv  
Here are 10 rows of results:  
GT=RM9 -> D6X: 0.0%  PY9: 1.0%  RM9: 96.3%  TK1: 2.7%  ZZ4: 0.0%  
GT=PY9 -> D6X: 0.0%  PY9:100.0% RM9: 0.0%   TK1: 0.0%  ZZ4: 0.0%  
GT=RM9 -> D6X: 0.0%  PY9: 0.8%  RM9: 88.9%  TK1: 10.3% ZZ4: 0.0%  
GT=D6X -> D6X: 89.3% PY9: 10.5% RM9: 0.2%   TK1: 0.0%  ZZ4: 0.0%  
GT=D6X -> D6X: 97.1% PY9: 2.8%  RM9: 0.0%   TK1: 0.0%  ZZ4: 0.1%  
GT=PY9 -> D6X: 0.0%  PY9:100.0% RM9: 0.0%   TK1: 0.0%  ZZ4: 0.0%  
GT=RM9 -> D6X: 0.4%  PY9: 46.3% RM9: 52.6%  TK1: 0.8%  ZZ4: 0.0%  
GT=D6X -> D6X: 98.7% PY9: 0.5%  RM9: 0.0%   TK1: 0.0%  ZZ4: 0.8%  
GT=RM9 -> D6X: 0.0%  PY9: 0.1%  RM9: 55.3%  TK1: 44.6% ZZ4: 0.0%  
GT=D6X -> D6X: 99.3% PY9: 0.6%  RM9: 0.0%   TK1: 0.0%  ZZ4: 0.1%  
*** Analysis ***  
302 data points analyzed, 257 correct (85.1% accuracy)  
Confusion:  
[[114 14 0 0 2]
 [ 12 76 7 0 0]
 [  3 0 56 2 0]
 [  0 0  2 9 0]
 [  3 0  0 0 2]]  
Wrote confusion matrix plot to confusion_gn.png  
*** Making a plot ****  
Saved to "confidence_gn.png".  
