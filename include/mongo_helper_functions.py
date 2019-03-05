import pymongo
import time
import logging
import pandas as pd
import datetime as dt
from pandas.io.json import json_normalize


def connectMongo(collection_name=None, db_name='cosine', host=None):
    if collection_name:
        return pymongo.MongoClient(host=host)[db_name][collection_name]
    else:
        return pymongo.MongoClient(host=host)[db_name]


def query(collection, pipeline, allowDiskUse=True, batchSize=100000):
    init = dt.datetime.now()
    df = json_normalize(
        list(collection.aggregate(
            pipeline,
            allowDiskUse=allowDiskUse,
            batchSize=batchSize
        ))
    )
    logging.info('Query duration: {}'.format(dt.datetime.now() - init))
    return df


def limit(n=0, pipeline=[]):
    pipeline = pipeline if pipeline else []
    if n:
        pipeline.append({'$limit': n})
    return pipeline


def project(pipeline=[], **kwargs):
    pipeline = pipeline if pipeline else []
    if kwargs:
        projection = {'_id': 0}
        projection.update(kwargs)
        pipeline.append({'$project': projection})
    return pipeline


def match(pipeline=[], **kwargs):
    pipeline = pipeline if pipeline else []
    if kwargs:
        pipeline.append({'$match': kwargs})
    return pipeline


def matchInterval(gte=None, lte=None):
    func = {}
    if gte:
        func.update({'$gte': gte})
    if lte:
        func.update({'$lte': lte})
    return func


def matchEpochFromDate(gte=None, lte=None):
    if gte:
        gte = time.mktime(gte.timetuple())
    if lte:
        lte = time.mktime(lte.timetuple())
    return matchInterval(gte, lte)


def convertEpochColToDatetime(df, epochCol='timestamp', offset='04:00:00'):
    if df[epochCol].dtype.name == 'int64':
        # fixed 4h based on time difference to created_date field.
        df[epochCol] = pd.to_datetime(df[epochCol], unit='s') - pd.Timedelta(offset)


def group(pipeline=[], **kwargs):
    pipeline = pipeline if pipeline else []
    if kwargs:
        if '_id' not in kwargs:
            group_query = {'_id': None}
        else:
            group_query = {}
        group_query.update(kwargs)
        pipeline.append({'$group': group_query})
    return pipeline


def prepareBulkUpdate(data, find_field, update_fields, **kwargs):
    return pymongo.operations.UpdateOne(
        {find_field: data[find_field]},
        {'$set': {update_field: data[update_field] for update_field in update_fields}},
        **kwargs
    )


def updateCollectionFromDataFrame(collection, df,
                                  bulk_func, func_args=(),
                                  ordered=False, **kwargs):
    init = dt.datetime.now()
    try:
        requests = df.apply(func=bulk_func, axis=1, args=func_args, **kwargs).tolist()
        res = collection.bulk_write(requests, ordered=ordered)
        logging.info('Update duration: {}'.format(dt.datetime.now() - init))
        return res
    except Exception as e:
        logging.error(e)
        return requests
