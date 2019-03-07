# Common scripts for all tasks

Description of Common Scripts.


### Mongo: Compute distributions of hashtags

```
pipeline = [
  {'$match': {'tweet.entities.hashtags': {$ne: []}}},
  {'$unwind': '$tweet.entities.hashtags'},
  {'$project': {hashtag:'$tweet.entities.hashtags.text'}},
  {'$group': {
      _id: '$hashtag',
      'count': {$sum: 1}
    }
  },
  {'$sort': {'count':-1}},
  {'$out':'tweets_count_hashtags'}
]
db.tweets.aggregate(pipeline)
```
