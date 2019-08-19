# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

from googletrans import Translator
translator = Translator()

import tweepy
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#import functools
from wordcloud import WordCloud
import re
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
#from string import punctuation

#from tweepy import Stream
#from tweepy import OAuthHandler
#from tweepy.streaming import StreamListener

consumer_key = 'pUJEX8B8HZ0YtWRk7o41057bZ'
consumer_secret = '3GCnmbCI3b5go4gznLr4g3eDE4h25mGkJT9F5QgSnCPxhZmS1B'
access_token = '132419103-9V5Zj1KJCwbQ5gt8J9hHd3p6cDgEnCdohpnyR03s'
access_token_secret = 'StaeieAgXDPF9jrA5kQmYjUre8y7QZwZcIVMZ6GBBsDwE'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.user_timeline('@FlyNAMAir', count=200, tweet_mode='extended')
for t in tweets:
    print(t.full_text)
    print()

def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw
    
def sentiment_analyzer_scores(text, engl=True):
    if engl:
        trans = text
    else:
        trans = translator.translate(text).text
    score = analyser.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    # remove ascii
    text = _removeNonAscii(text)
    # to lowecase
    text = text.strip()
    return text

def clean_lst(lst):
    lst_new = []
    for r in lst:
        lst_new.append(clean_text(r))
    return lst_new

def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^\w\s+]", " ")
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst


stop_words = []
f = open('D:\ProjectDTS\stopwordId.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))
    
additional_stop_words = ['t', 'will', 'yang', 'loh', 'nya']
stop_words += additional_stop_words

#print(len(stop_words))


#sentiment_analyzer_scores(tw_namair[16])
def anl_tweets(lst, title='Tweets Sentiment', engl=True ):
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(sents,kde=False,bins=3)
    ax.set(xlabel='Negative                Neutral                 Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents


def word_cloud(wd_list):
    stopwords = stop_words + list('stopwords')
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=21,
        colormap='jet',
        max_words=50,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");

user_id = 'FlyNAMAir'
count=200
tw_namair = list_tweets(user_id, count)
tw_namair = clean_tweets(tw_namair)
#tw_namair[5]
#sentiment_analyzer_scores(tw_namair[5])
tw_namair_sent = anl_tweets(tw_namair, user_id) 
word_cloud(tw_namair)