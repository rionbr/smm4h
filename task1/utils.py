import sys
sys.path.insert(0, '../include')

from datetime import date, datetime
import pandas as pd
from termdictparser import Sentence, TermDictionaryParser
from mongo_helper_functions import connectMongo, match, query, project, limit


def load_task1_classes():
    """ Loads Classes (y) for Task 1"""
    # Load Positive/Negative sets for Task 1
    ldfC = []
    for i in [1, 2, 3]:
        file = '../data/task1/training_set_{:d}_ids.txt'.format(i)
        dft = pd.read_csv(file, sep='\t', names=['tweet_id', 'user_id', 'category'])
        ldfC.append(dft)
    dfC = pd.concat(ldfC, axis=0)
    dfC['tweet_id_str'] = dfC['tweet_id'].astype(str)
    dfC['user_id_str'] = dfC['user_id'].astype(str)
    dfC.set_index('tweet_id_str', inplace=True)
    return dfC


class DictionaryParser(object):
    """docstring for DictionaryParser."""

    def __init__(self, db, type):
        self.db = db
        self.type = type

        # Build Vocabulary
        pipeline = match(type=self.type)
        pipeline = project(pipeline=pipeline, _id=1, token=1, parent=1, type=1)
        df = query(collection=self.db['dict_20180706'], pipeline=pipeline)
        # Some tokens have multiple hits (Drug products with multiple compounds)
        dfg = df.groupby('token').agg({'_id': lambda x: tuple(x)})
        dfg = dfg.reset_index()
        df = df.set_index('_id')
        self.dict_token = df['token'].to_dict()
        self.dict_parent = df['parent'].to_dict()
        self.dict_type = df['type'].to_dict()

        # Build Term Parser
        self.tdp = TermDictionaryParser()

        def build_sentences(row):
            return Sentence(id=row['_id'], text=row['token']).preprocess(lower=True).re_tokenize(re=None).lemmatize(pos='v')

        dfg['sentences'] = dfg.apply(build_sentences, axis=1)

        self.tdp.build_vocabulary(dfg['sentences'].values)


Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y, 1, 1), date(Y, 3, 20))),
           ('spring', (date(Y, 3, 21), date(Y, 6, 20))),
           ('summer', (date(Y, 6, 21), date(Y, 9, 22))),
           ('autumn', (date(Y, 9, 23), date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21), date(Y, 12, 31)))]


def calc_season_from_datetime(now):
    """
    Retrieved the season (for northern hemisphere) based on a datetime.
    """
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=2000)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def count_pattern_in_list(lst, seq):
    """
    Counts a specific pattern of objects within a list.

    Usage:
        count_pattern_in_list([1,0,1,2,0,1], [0,1]) = 2
    """
    count = 0
    len_seq = len(seq)
    upper_bound = len(lst) - len_seq + 1
    for i in range(upper_bound):
        if lst[i:i + len_seq] == seq:
            count += 1
    return count


if __name__ == '__main__':

    db = connectMongo(db_name='smm4h', host='angst.soic.indiana.edu')

    DPDrug = DictionaryParser(db=db, type='Drug')
    print(DPDrug)

    s1 = u"I am having FLuoXeTiNe high ADHD, ADHD! #LINHAÇA! losing weight but not because I'm doing a weight lossing diet, linhaça. It's my Nerve block back again with FLuoxetine. Perhaps I need to take some multi vitamins. huh nerve. #first. #second."
    s1 = Sentence(s1).preprocess(lower=True, remove_hash=True).re_tokenize(re=None).lemmatize(pos='v').match_tokens(parser=DPDrug.tdp)

    for match in s1.matches:
        print(match)
