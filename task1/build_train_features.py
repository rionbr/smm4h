# coding=utf-8
# Author: Rion Brattig Correia
# Date: April 03, 2019
#
# Description:
# Build features and saves them on Mongo Collection.
#
#
import sys
sys.path.insert(0, '../include')
from mongo_helper_functions import connectMongo, match, query, project, limit
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import nltk
import time
from itertools import chain
from collections import Counter
from termdictparser import Sentences, TermDictionaryParser
from utils import DictionaryParser, load_task1_classes, count_pattern_in_list


if __name__ == '__main__':


    # Load Positive/Negative sets for Task 1
    dfC = load_task1_classes()
    dict_number_of_positive_cases = dfC.loc[ (dfC['category']==1) , 'user_id'].value_counts().to_dict()
    dict_category = dfC['category'].to_dict()

    # Connect to Mongo
    db = connectMongo(db_name='smm4h', host='angst.soic.indiana.edu')

    # Dictionary Parser
    DP = DictionaryParser()
    # Season Calculator
    SC = SeasonCalculator()

    # Load tweets
    pipeline = match(pipeline=[], task_one=True)
    pipeline = project(pipeline=pipeline, _id=1, task_one=1, tweet=1, datetime=1)
    pipeline = limit(n=100, pipeline=pipeline)
    dfP = query(collection=db['tweets'], pipeline=pipeline)
    dfP.set_index('_id', inplace=True)

    # Load timelines
    pipeline = [{'$match': {'tweet.user.id_str': {'$in': dfC['user_id'].tolist()}}}]
    pipeline = project(pipeline=pipeline, _id=1, task_one=1, tweet=1, datetime=1)
    pipeline = limit(n=10, pipeline=pipeline)
    dfT = query(collection=db['timelines'], pipeline=pipeline)
    dfT.set_index('_id', inplace=True)

    # Results
    dfI = pd.DataFrame(index=dfP.index)

    # User Features
    dfI['y'] = dfI.index.map(lambda x: dict_category[x])

    # User Features
    dfI['user.number_of_friends'] = dfP['tweet.user.friends_count']
    dfI['user.log(number_of_friends)'] = np.log(dfP['tweet.user.friends_count'])
    dfI['user.number_of_followers'] = dfP['tweet.user.followers_count']
    dfI['user.log(number_of_followers)'] = np.log(dfP['tweet.user.followers_count'])
    dfI['user.friends/followers'] = dfP['tweet.user.friends_count'] / dfP['tweet.user.followers_count']
    dfI['user.number_of_tweets'] = dfP['tweet.user.statuses_count']
    dfI['user.log(number_of_tweets)'] = np.log(dfP['tweet.user.statuses_count'])

    def calc_positive_cases(_id):
        if dict_number_of_positive_cases.get(_id) is not None:
            return dict_number_of_positive_cases.get(_id)
        else:
            return np.nan
    dfI['user.number_of_positive_cases'] = dfI.index.map(calc_positive_cases)
    dfI['user.ratio_positive_negative_cases'] = dfI['user.number_of_positive_cases'] / (dfI['user.number_of_positive_cases'] - dfI['user.number_of_tweets'])

    # Load User Timeline
    tmatch = {'tweet.user.id_str': r['_id']}
    tfields = {'tweet': 1, 'datetime': 1}
    tresults = db['timelines'].find(tmatch, fields)

    # Temporal Features
    dfI['temp.hour_of_day'] = dfP['datetime'].dt.hour
    dfI['temp.day_of_week'] = dfP['datetime'].dt.strftime('%A').str.lower()  # Or simply ".dt.weekday"
    dfI['temp.season'] = dfP['datetime'].map(calc_season_from_datetime)

    # Textual Features
    def textual_parse_tweet(text):

        sentence_obj = Sentences(text).preprocess(lower=True, remove_mentions=True, remove_url=True).tokenize()
        tagged_sentences = [nltk.pos_tag(x, lang='eng', tagset='universal') for x in sentence_obj.tokens_sentences]
        flat_word, flat_tags = zip(*chain(*tagged_sentences))
        counted_tags = Counter(flat_tags)
        sT = pd.Series(counted_tags, index=['text.number_of_({:s})'.format(tag) for tag in ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.']]).fillna(0)
        sR = pd.Series({
            'text.length_of_text': len(text),
            'text.number_of_words': len([s for ss in sentence_obj.tokens_sentences for s in ss]),
            'text.number_of_(NOUN+VERB+ADJ)': count_pattern_in_list(flat_tags, ['NOUN', 'VERB', 'ADJ'])
        })

        return pd.concat([sR,sT], axis=0)

    dftemp = dfP['tweet.text'].apply(textual_parse_tweet)

    updateCollectionFromDataFrame(
        collection=task_1_train_features,
        df=dfI,
        bulk_func=prepareBulkUpdate,
        find_field='root_id',
        update_fields=['root_user','root_time','subreddit'],
        upsert=True
    )
