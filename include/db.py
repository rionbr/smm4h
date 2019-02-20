# coding=utf-8
# Author: Rion B Correia
# Date: Feb 19, 2019
#
# Description:
# Database Connection for Mongo
#
#
import os
import json
# Try: Python2; Except: Python3
try:
    from urllib import quote_plus
except ImportError:
    from urllib.parse import quote_plus
from pymongo import MongoClient


with open(os.path.join(os.path.dirname(__file__), 'db_config.json')) as f:
    CONFIGS = json.load(f)


def connectToMongoDB(server, verbose=False, *args, **kwargs):
    """Connect ot Mongo DB"""
    if server not in CONFIGS:
        raise ValueError('Database server `%s` not defined in `db_config.json`.' % (server))
    else:
        CONFIG = CONFIGS[server].copy()
    # Has Authentication
    if 'user' in CONFIG and 'password' in CONFIG:
        CONFIG['user'] = quote_plus(CONFIG['user'])
        CONFIG['password'] = quote_plus(CONFIG['password'])
        url = 'mongodb://%(user)s:%(password)s@%(host)s/?authMechanism=MONGODB-CR&authSource=admin' % CONFIG
    else:
        url = 'mongodb://%(host)s' % CONFIG
    client = MongoClient(url, *args, **kwargs)
    return client
