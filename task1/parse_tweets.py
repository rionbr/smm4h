# coding=utf-8
# Author: Rion Brattig Correia
# Date: Feb 19, 2019
#
# Description:
# Parse tweets: a) matches dictionaries
#
#
import sys
sys.path.insert(0, '../include')
from mongo_helper_functions import connectMongo, query, project, limit
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import nltk
import time
from termdictparser import Sentences, TermDictionaryParser
import utils

if __name__ == '__main__':

    # Load Positive/Negative sets for Task 1
    dfC = utils.load_task1_classes()
    dict_category = dfC['category'].to_dict()

    # Connect to Mongo
    db = connectMongo(db_name='smm4h', host='angst.soic.indiana.edu')

    # Load Dictionary (dict_20180706)
    print('--- Loading dict_20180706 Dict ---')
    pipeline = project(_id=1, token=1, parent=1, type=1)
    # pipeline = limit(n=10, pipeline=pipeline)  # debug
    dfD = query(collection=db['dict_20180706'], pipeline=pipeline)
    # Some tokens have multiple hits (Drug products with multiple compounds)
    dfDg = dfD.groupby('token').agg({'_id': lambda x: tuple(x)})
    dfDg = dfDg.reset_index().set_index('_id')
    dfD = dfD.set_index('_id')
    dict_token = dfD['token'].to_dict()
    dict_parent = dfD['parent'].to_dict()
    dict_type = dfD['type'].to_dict()

    # Build Term Parser
    tdp = TermDictionaryParser()
    # Select columns to pass to parser
    list_tuples = list(dfDg['token'].str.lower().items())
    # Build Parser Vocabulary
    tdp.build_vocabulary(list_tuples)

    # Load Dictionary (national_drug_product)
    print('--- Loading national_drug_product Dict ---')
    pipeline = project(_id=1, proprietaryname=1, substancename=1, nonproprietaryname=1, pharm_classes=1)
    # pipeline = limit(n=10, pipeline=pipeline)  # debug
    dfD2 = query(collection=db['national_drug_product'], pipeline=pipeline)
    # Remove duplicates and NaN
    dfD2 = dfD2.drop_duplicates(subset=['proprietaryname']).dropna(subset=['proprietaryname'])
    dfD2 = dfD2.set_index('_id')
    dict2_nonpropname = dfD2['nonproprietaryname'].to_dict()
    dict2_pharmclass = dfD2['pharm_classes'].to_dict()
    dict2_substance = dfD2['substancename'].to_dict()

    # Build Term Parser
    tdp2 = TermDictionaryParser()
    # Select columns to pass to parser
    list_tuples = list(dfD2['proprietaryname'].str.lower().items())
    # Build Parser Vocabulary
    tdp2.build_vocabulary(list_tuples)

    # Load tweets (not using mongo_helper_functions deliberately)
    print('--- Matching Tweets ---')
    match = {'task_one': True}
    fields = {'task_one': 1, 'tweet': 1, 'datetime': 1}
    results = db['tweets'].find(match, fields)  # .limit(1)
    inserts = list()
    for r in results:
        obj = dict()
        # General Info
        obj['_id'] = str(r['_id'])
        t = r['tweet']
        u = t['user']
        obj['class'] = dict_category[int(r['_id'])]
        # User Info
        obj['user_id'] = u['id_str']
        obj['user_location'] = u['location']
        obj['user_favourites_count'] = u['favourites_count']
        obj['user_friends_count'] = u['friends_count']
        obj['user_description'] = u['description']
        obj['user_followers_count'] = u['followers_count']
        obj['user_lang'] = u['lang']
        obj['user_timezone'] = u['time_zone']
        obj['user_statuses_count'] = u['statuses_count']
        obj['user_name'] = u['name']
        obj['user_created_at'] = u['created_at']
        # Tweet Info
        obj['post_datetime'] = r['datetime']
        obj['post_text'] = t['text']
        obj['post_retweeted'] = t['retweeted']
        obj['post_coordinates'] = t['coordinates']
        obj['post_hashtags'] = [x['text'] for x in t['entities']['hashtags']]
        obj['post_symbols'] = [x['text'] for x in t['entities']['symbols']]
        obj['post_urls'] = [x['expanded_url'] for x in t['entities']['urls']]
        obj['post_user_mentions'] = [x['id_str'] for x in t['entities']['user_mentions']]
        obj['post_place'] = t['place']

        # POS Tagging (Parts-Of-Speech)
        s = Sentences(t['text']).preprocess(lower=True, remove_mentions=True, remove_url=True).tokenize()
        obj['post_text_pp'] = s.text_pp
        obj['post_text_pos_tag'] = [nltk.pos_tag(x) for x in s.tokens_sentences]
        # Dictionary Matches
        matches = []
        s = s.match_tokens(parser=tdp)
        if s.has_match():
            for match in s.get_unique_matches():
                for mid in match.id:
                    matches.append({
                        'id': mid,
                        'token': dict_token[mid],
                        'parent': dict_parent[mid],
                        'type': dict_type[mid]
                    })
        obj['matches_dict'] = matches

        matches2 = []
        s = s.match_tokens(parser=tdp2)
        if s.has_match():
            for match in s.get_unique_matches():
                mid = match.id
                matches2.append({
                    'id': mid,
                    'nonpropname': dict2_nonpropname[mid],
                    'pharm_classes': dict2_pharmclass[mid],
                    'substance_name': dict2_substance[mid]
                })
        obj['matches_ndp'] = matches2

        inserts.append(obj)

    # Insert into Mongo
    print('--- Inserting to Mongo ---')
    if len(inserts):
        inserts_size = len(inserts)
        chunk_size = 1000
        if inserts_size > chunk_size:
            for i in range(0, inserts_size, chunk_size):
                inserts_chunk = inserts[i:i + chunk_size]
                # Try-Retry function for Mongo
                for t in range(10):
                    try:
                        db['tweets_parsed_task1'].insert_many(inserts_chunk, ordered=False)
                        break
                    except pymongo.errors.AutoReconnect:
                        time.sleep(pow(2, t))

        else:
            db['tweets_parsed_task1'].insert_many(inserts, ordered=False)
