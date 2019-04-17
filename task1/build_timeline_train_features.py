# coding=utf-8
# Author: Rion Brattig Correia
# Date: April 17, 2019
#
# Description:
# Build timeline features and updates them on the Mongo Collection.
#
#
import sys
sys.path.insert(0, '../include')
sys.path.insert(0, '../')
from mongo_helper_functions import connectMongo, match, query, project, limit, prepareBulkUpdate, updateCollectionFromDataFrame
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from termdictparser import Sentence
from utils import DictionaryParser
from sentiment.sentiment import Sentiment
from joblib import Parallel, delayed, wrap_non_picklable_objects


def timeline_textual_features(text):
    sentence_obj = Sentence(text).preprocess(lower=True, remove_mentions=True, remove_url=True).tokenize()

    # Match tokens to dictionary
    matches_drug = sentence_obj.match_tokens(DPDrug.tdp).get_matches()
    matches_medicalterms = sentence_obj.match_tokens(DPMedTerm.tdp).get_matches()
    matches_naturalproducts = sentence_obj.match_tokens(DPNatProd.tdp).get_matches()

    sR = pd.Series({
        'timeline_length_text': len(text),
        'timeline_number_words': len([s for ss in sentence_obj.tokens_sentences for s in ss]),
        'timeline_number_(Drugs)': len(matches_drug),
        'timeline_number_(MedicalTerms)': len(matches_medicalterms),
        'timeline_number_(NaturalProducts)': len(matches_naturalproducts),
    })
    sR = sR[sR > 0]  # Remove Zeros
    return sR


def timeline_features(user_id):
    global db
    # Load timeline
    pipeline = [{'$match': {'tweet.user.id_str': user_id}}]
    pipeline = project(pipeline=pipeline, **{'_id': 1, 'tweet.full_text': 1})
    pipeline = limit(n=None, pipeline=pipeline)
    dfT = query(collection=db['timelines'], pipeline=pipeline)
    if len(dfT) > 0:
        dftexttimeline = dfT['tweet.full_text'].apply(timeline_textual_features)
        return dftexttimeline.sum(axis='index').astype('int').to_dict()
    else:
        return dict


if __name__ == '__main__':

    # Connect to Mongo
    db = connectMongo(db_name='smm4h', host='angst.soic.indiana.edu')

    # Dictionary Parsers
    print('> Loading Dictionaries')
    DPDrug = DictionaryParser(db=db, type='Drug')
    DPMedTerm = DictionaryParser(db=db, type='Medical term')
    DPNatProd = DictionaryParser(db=db, type='Natural product')

    # Sentiment
    Sent = Sentiment()

    # Load tweets
    print('> Loading Tweets')
    pipeline = match(pipeline=[], task_one=True)
    pipeline = project(pipeline=pipeline, _id=1, tweet=1)
    pipeline = limit(n=None, pipeline=pipeline)
    dfP = query(collection=db['tweets'], pipeline=pipeline)
    dfP.set_index('_id', inplace=True, drop=False)
    post_ids = dfP['_id'].tolist()
    user_ids = dfP['tweet.user.id_str'].tolist()
    # Results
    dfI = pd.DataFrame(index=dfP.index.values)

    # Timeline Features
    print('> Timeline features')

    # Parallel Processing of timelines (also slow and can't pickle the worker)
    # rlist = Parallel(n_jobs=6, verbose=10, prefer='threads')(delayed(timeline_features)(user_id, db) for user_id in user_ids)

    # Serial Processing of timelines
    i, n = 0, len(user_ids)
    rlist = list()
    for user_id in user_ids:
        i += 1
        if (i % 100 == 0):
            print(". calculating timeline: {:d} of {:d} ({:.2%})".format(i, n, i / n))
        rdict = timeline_features(user_id)
        rlist.append(rdict)

    # Update Mongo
    print('> Updating collection')
    i = 0
    n = len(post_ids)
    for post_id, fieldvalues in zip(post_ids, rlist):
        i += 1
        if (i % 10 == 0):
            print(". updating record: {:d} of {:d} ({:.2%})".format(i, n, i / n))

        filter = {'_id': post_id}
        update = {'$set': {field: value for field, value in fieldvalues.items()}}
        db['task_1_train_features'].update_one(filter, update)
