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
from termdictparser import Sentences, TermDictionaryParser

if __name__ == '__main__':

    # Connect to Mongo
    db = connectMongo(db_name='smm4h', host='angst.soic.indiana.edu')

    # Load Dictionary
    pipeline = project({'token': 1})
    pipeline = limit(10)  #debug
    dfD = query(collection=db['dict_20180706'], pipeline=pipeline)

    print(dfD.head())

    # Build Term Parser
    tdp = TermDictionaryParser()
    # Select columns to pass to parser
    list_tuples = list(dfD['token'].str.lower().items())
    # Build Parser Vocabulary
    tdp.build_vocabulary(list_tuples)

    # Load tweets (not using mongo_helper_functions deliberately)
    match = {}
    fields = {'task_one': 1, 'task_two': 1, 'task_three': 1, 'task_four': 1, 'tweet': 1, 'datetime': 1}
    results = db['tweets'].find(match, fields).limit(1)
    print(results)
    for r in results:
        tid = r['_id']
        task1 = r['task_one']
        task2 = r['task_two']
        task3 = r['task_three']
        task4 = r['task_four']
        t = r['tweet']
        datetime = r['datetime']
        text = ''

    # RION STOPPED HERE.

    s = Sentences(text).preprocess(lower=True).tokenize().match_tokens(parser=tdp)
    if s.has_match():
        mj = {
            '_id': tid,
            'created_time': date_publication,
            'matches': []
        }
        mj.update( types ) # include

        for match in s.get_unique_matches():
            for mid in match.id:
                mj['matches'].append({
                    'id': mid,
                    'id_parent': dict_id_parent[mid],
                    'token': dict_token[mid],
                    'parent': dict_parent[mid],
                    'type': dict_type[mid]
                })

        try:
            db['tweets_parsed'].insert_one(mj)
        except ValueError as error:
            print("Error! Args: '{:s}'".format(error.args))
        else:
            pass
            # print 'No matches (PMID: {:d})'.format(pmid)
