# coding=utf-8
# Author: Rion B Correia
# Date: Feb 19, 2019
#
# Description:
# Exports Rion's Dictionary from MySQL and imports into the MongoDB.
# Note this script doesn't run stand-alone it needs to be ran from inside ddi_project.
# Rion can re-run this script if necessary.
#
import sys
sys.path.insert(0, '../../include')
sys.path.insert(0, '../../../include')
import db_init_ddi_project as db
import pandas as pd


if __name__ == '__main__':

    dicttimestamp = '20180706'

    print('--- Loading MySQL dictionary (%s)---' % dicttimestamp)
    engine = db.connectToMySQL(server='mysql_ddi_dictionaries')
    tablename = 'dict_%s' % (dicttimestamp)
    sql = """SELECT
        d.id,
        IFNULL(d.id_parent,d.id) AS id_parent,
        d.dictionary,
        d.token,
        IFNULL(p.token, d.token) as parent,
        d.type,
        d.source,
        d.id_original,
        IFNULL(p.id_original, d.id_original) as id_original_parent
        FROM %s d
        LEFT JOIN %s p ON d.id_parent = p.id
        WHERE d.enabled = True""" % (tablename, tablename)
    dfD = pd.read_sql(sql, engine, index_col='id')
    dfD = dfD.reset_index()

    mongo_mention, _ = db.connectToMongoDB(server='mongo_tweetline', db='smm4h')

    for i, row in dfD.iterrows():
        row = row.to_dict()
        row['_id'] = row.pop('id')
        mongo_mention['dict_{:s}'.format(dicttimestamp)].insert_one(row)
