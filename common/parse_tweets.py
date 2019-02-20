# coding=utf-8
# Author:
# Date: Feb 19, 2019
#
# Description:
# Parse tweets. Just a DB test for now.
#
#
import sys
sys.path.insert(0, '../include')
from db import connectToMongoDB
import pandas as pd

if __name__ == '__main__':

    # Connect to Mongo
    engine = connectToMongoDB(server='mongo_on_angst')
    db = engine['smm4h']

    # Load Dictionary
    query = {}
    fields = {'_id': 1, 'token': 1}
    cursor = db['dict_20180706'].find(query, fields)
    dfD = pd.DataFrame(list(cursor))

    print(dfD.head())

    # More to come...
