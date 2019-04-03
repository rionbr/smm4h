from datetime import date, datetime
import pandas as pd
from termdictparser import Sentences, TermDictionaryParser
from mongo_helper_functions import connectMongo, query, project, limit


def load_task1_classes():
    """ Loads Classes (y) for Task 1"""
    # Load Positive/Negative sets for Task 1
    ldfC = []
    for i in [1, 2, 3]:
        file = '../data/task1/training_set_{:d}_ids.txt'.format(i)
        dft = pd.read_csv(file, sep='\t', names=['tweet_id', 'user_id', 'category'])
        ldfC.append(dft)
    dfC = pd.concat(ldfC, axis=0)
    dfC['tweet_id_str'] = dfC['tweet_id'].astype('string')
    dfC.set_index('tweet_id_str', inplace=True)
    return dfC


class DictionaryParser(object):
    """docstring for DictionaryParser."""

    def __init__(self, db):
        # Load Dictionary (dict_20180706)
        print('--- Loading dict_20180706 Dict ---')
        pipeline = project(_id=1, token=1, parent=1, type=1)
        # pipeline = limit(n=10, pipeline=pipeline)  # debug
        dfD = query(collection=db['dict_20180706'], pipeline=pipeline)
        # Some tokens have multiple hits (Drug products with multiple compounds)
        dfDg = dfD.groupby('token').agg({'_id': lambda x: tuple(x)})
        dfDg = dfDg.reset_index().set_index('_id')
        dfD = dfD.set_index('_id')
        self.dict_token = dfD['token'].to_dict()
        self.dict_parent = dfD['parent'].to_dict()
        self.dict_type = dfD['type'].to_dict()

        # Build Term Parser
        self.tdp = TermDictionaryParser()
        # Select columns to pass to parser
        list_tuples = list(dfDg['token'].str.lower().items())
        # Build Parser Vocabulary
        self.tdp.build_vocabulary(list_tuples)

    def match(self, text):
        sentences = Sentences(text).preprocess(lower=True, remove_mentions=True, remove_url=True).tokenize()
        matches = []
        sentences = sentences.match_tokens(parser=self.tdp)
        if sentences.has_match():
            for match in sentences.get_unique_matches():
                for mid in match.id:
                    matches.append({
                        'id': mid,
                        'token': self.dict_token[mid],
                        'parent': self.dict_parent[mid],
                        'type': self.dict_type[mid]
                    })
        return matches


Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y, 1, 1), date(Y, 3, 20))),
           ('spring', (date(Y, 3, 21), date(Y, 6, 20))),
           ('summer', (date(Y, 6, 21), date(Y, 9, 22))),
           ('autumn', (date(Y, 9, 23), date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21), date(Y, 12, 31)))]


def calc_season_from_datetime(now):
    """
    Retrieved the season (northern hemisphere) based on a datetime.
    """
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=2000)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def count_pattern_in_list(lst, seq):
    """
    Counts a specific pattern of objects within a list.
    """
    count = 0
    len_seq = len(seq)
    upper_bound = len(lst) - len_seq + 1
    for i in range(upper_bound):
        if lst[i:i + len_seq] == seq:
            count += 1
    return count
