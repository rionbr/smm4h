# coding=utf-8
# Author: Rion Brattig Correia
# Date: April 03, 2019
#
# Description:
# Build post features and saves them on Mongo Collection.
#
#
import sys
sys.path.insert(0, '../include')
sys.path.insert(0, '../')
from mongo_helper_functions import connectMongo, match, query, project, limit
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import nltk
from itertools import chain
from collections import Counter
from termdictparser import Sentence
from utils import DictionaryParser, load_task1_classes, count_pattern_in_list, calc_season_from_datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment.sentiment import Sentiment

if __name__ == '__main__':

    # Change for building features for 'train' or 'test'
    train_or_test = 'test'

    # Load Classes (train only)
    if train_or_test == 'train':
        # Load Positive/Negative sets for Task 1
        print('> Positive/Negative Classes')
        dfC = load_task1_classes()
        dict_number_of_positive_cases = dfC.loc[(dfC['category'] == 1), 'user_id'].value_counts().to_dict()
        dict_category = dfC['category'].to_dict()

    # Load Predict (test only)
    if train_or_test == 'test':
        dfT = pd.read_csv('../data/task1/testDataST1_participants.tsv', sep='\t', names=['_id', 'task_id', 'tweet.text'], dtype={'_id': 'str'}).set_index('_id')

    # Connect to Mongo
    if train_or_test == 'train':
        db_name = 'smm4h'
    elif train_or_test == 'test':
        db_name = 'smm4h_final'

    db_dict = connectMongo(db_name='smm4h', host='angst.soic.indiana.edu')
    db = connectMongo(db_name=db_name, host='angst.soic.indiana.edu')

    # Dictionary Parsers
    print('> Loading Dictionaries')
    DPDrug = DictionaryParser(db=db_dict, type='Drug')
    DPMedTerm = DictionaryParser(db=db_dict, type='Medical term')
    DPNatProd = DictionaryParser(db=db_dict, type='Natural product')

    # Sentiment
    Sent = Sentiment()

    # Load tweets
    print('> Loading Tweets')
    pipeline = match(pipeline=[], task_one=True)
    pipeline = project(pipeline=pipeline, _id=1, task_one=1, tweet=1, datetime=1)
    pipeline = limit(n=None, pipeline=pipeline)
    dfP = query(collection=db['tweets'], pipeline=pipeline)
    dfP['_id'] = dfP['_id'].astype('str')
    dfP.set_index('_id', inplace=True, drop=False)

    if train_or_test == 'test':
        dfP.drop(labels=['tweet.text', '_id'], axis='columns', inplace=True)
        dfP = pd.concat([dfT, dfP], axis='columns', verify_integrity=True, sort=False)

    # Results (remember to keep '_id' also a column for the Mongo Insert)
    dfI = pd.DataFrame({'_id': dfP.index.values}, index=dfP.index.values)

    # Class
    if train_or_test == 'train':
        dfI['y'] = dfI.index.map(lambda x: dict_category[x])

    # User Features
    dfI['user_number_friends'] = dfP['tweet.user.friends_count']
    dfI['user_log(number_friends)'] = np.log(dfP['tweet.user.friends_count'] + 1)
    dfI['user_number_followers'] = dfP['tweet.user.followers_count']
    dfI['user_log(number_followers)'] = np.log(dfP['tweet.user.followers_count'] + 1)
    dfI['user_ratio_friends_followers'] = dfP['tweet.user.friends_count'] / dfP['tweet.user.followers_count']
    dfI['user_number_tweets'] = dfP['tweet.user.statuses_count']
    dfI['user_log(number_tweets)'] = np.log(dfP['tweet.user.statuses_count'] + 1)

    """
    def calc_positive_cases(_id):
        if dict_number_of_positive_cases.get(_id) is not None:
            return dict_number_of_positive_cases.get(_id)
        else:
            return np.nan
    dfI['user_number_positive_cases'] = dfI.index.map(calc_positive_cases)
    dfI['user_ratio_positive_negative_cases'] = dfI['user_number_positive_cases'] / (dfI['user_number_positive_cases'] - dfI['user_number_tweets'])
    """

    # Temporal Features
    dfI['temp_hour_of_day'] = dfP['datetime'].dt.hour
    dfI['temp_day_of_week'] = dfP['datetime'].dt.strftime('%A').str.lower()  # Or simply ".dt.weekday"
    dfI.loc[ (dfI['temp_day_of_week'] == 'nat'), 'temp_day_of_week'] = pd.NaT  # make sure weekday is NaT and not 'nat'
    dfI['temp_season'] = dfP['datetime'].map(calc_season_from_datetime)

    # Sentiment Features
    def sentiment_parse_tweet(text):
        sent_dict = Sent.calculate_average_score(text)
        sS = pd.Series({"sent_" + k: sent_dict[k] for k in sent_dict})
        sS = sS[sS > 0]  # Remove Zeros
        return sS

    dfsent = dfP['tweet.text'].apply(sentiment_parse_tweet)

    # Textual Features
    def tweet_textual_features(text):

        sentence_obj = Sentence(text).preprocess(lower=True, remove_mentions=True, remove_url=True).tokenize()
        tagged_sentences = [nltk.pos_tag(x, lang='eng', tagset='universal') for x in sentence_obj.tokens_sentences]
        flat_word, flat_tags = zip(*chain(*tagged_sentences))
        counted_tags = Counter(flat_tags)
        # Count POS tags
        sT = pd.Series(counted_tags, index=['text_number_of_({:s})'.format(tag) for tag in ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', 'pct']]).fillna(0)
        sT = sT[sT > 0]  # Remove all zeros

        # Match tokens to dictionary
        matches_drug = sentence_obj.match_tokens(DPDrug.tdp).get_matches()
        matches_medicalterms = sentence_obj.match_tokens(DPMedTerm.tdp).get_matches()
        matches_naturalproducts = sentence_obj.match_tokens(DPNatProd.tdp).get_matches()

        # Parent names for Dictionary terms
        parents_drug = [DPDrug.dict_parent[id] for match in matches_drug for id in match.id]
        parents_medicalterms = [DPMedTerm.dict_parent[id] for match in matches_medicalterms for id in match.id]
        parents_naturalproducts = [DPNatProd.dict_parent[id] for match in matches_naturalproducts for id in match.id]
        parent_text = ' '.join(parents_drug + parents_medicalterms + parents_naturalproducts)

        sR = pd.Series({
            'post_length_text': len(text),
            'post_number_words': len([s for ss in sentence_obj.tokens_sentences for s in ss]),
            'post_number_(NOUN+VERB+ADJ)': count_pattern_in_list(flat_tags, ['NOUN', 'VERB', 'ADJ']),
            'post_number_(Drugs)': len(matches_drug),
            'post_number_(MedicalTerms)': len(matches_medicalterms),
            'post_number_(NaturalProducts)': len(matches_naturalproducts),
            'parent_text': parent_text,  # this is not a feature, it gets removed next.
        })
        sR = sR[sR != 0]  # Remove Zeros

        # Return a Series to each row of a new DataFrame
        return pd.concat([sR, sT], axis='index')

    print('> Tweet textual features')
    dftextpost = dfP['tweet.text'].apply(tweet_textual_features)
    dfP['tweet.parent_text'] = dftextpost['parent_text']  # Parent Text
    dftextpost.drop(['parent_text'], axis='columns', inplace=True)

    # TF-IDF
    print('> TF-IDF for tweet')
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 1), max_df=0.9, min_df=5, max_features=1000, binary=False)
    X = tfidf.fit_transform(dfP['tweet.text'].values)
    tfidf_feature_names = ['post_tfidf_(' + name + ')' for name in tfidf.get_feature_names()]
    dftfidf = pd.SparseDataFrame(X, columns=tfidf_feature_names, index=dfP.index)

    # TF-IDF (for parent text)
    print('> TF-IDF for parent text')
    X = tfidf.fit_transform(dfP['tweet.parent_text'].values)
    tfidf_feature_names = ['post_tfidf_parent_(' + name + ')' for name in tfidf.get_feature_names()]
    dftfidf_parent = pd.SparseDataFrame(X, columns=tfidf_feature_names).set_index(dfP.index)

    # Final concat
    dfI = pd.concat([
                    dfI,  # Base features
                    dfsent,  # Sentiment features
                    dftextpost,  # Textual features
                    dftfidf,  # TF-IDF features
                    dftfidf_parent  # TF-IDF features on parent terms
                    ], sort=False, axis='columns')

    # Round all float columns to 4 digits
    dfI = dfI.round(6)

    # Insert to Mongo
    print('> Drop Collection')
    col = 'task_1_{:s}_features'.format(train_or_test)
    db[col].drop()

    print('> Inserting collection')
    # Converts DF columns (index is discarded) into list of dicts, removing NaNs
    inserts = [{k: v for k, v in m.items() if pd.notnull(v)} for m in dfI.to_dict(orient='rows')]
    db[col].insert_many(inserts, ordered=False)
