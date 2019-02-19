import pymongo

# Creates indexes in both the tweet and timeline collections on the tweet_id and the user_id fields.

mongo = pymongo.MongoClient(connect=False)
db = mongo.smm4h

tweets = db.tweets
timelines = db.timelines

tweets.create_index("tweet.id_str")
tweets.create_index("tweet.user.id_str")

timelines.create_index("tweet.id_str")
timelines.create_index("tweet.user.id_str")
