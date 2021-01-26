import flask
from flask import request, jsonify
import numpy as np
import pandas as pd
import json
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

from newsapi import NewsApiClient
api = NewsApiClient(api_key='0924f039000046a99a08757a5b122a4c')

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/stock/<ticker>', methods=['GET'])
def stock(ticker):
    myList = api.get_everything(q=ticker)['articles'][:100]
    news_titles = []

    for x in myList:
        news_titles.append(x['title'])
    df = pd.DataFrame([], columns = ['TKR', 'Headline'])
    df['Headline'] = news_titles
    df['TKR'] = ticker

    vader = SentimentIntensityAnalyzer()
    scores = df['Headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = df.join(scores_df, rsuffix='_right')
    x = parsed_and_scored_news[abs(parsed_and_scored_news.compound) > .1] # take out neutral articles, as the reduce sentiment score

    return {
        "score": x['compound'].mean(),
        "articles": myList
    }

app.run()

