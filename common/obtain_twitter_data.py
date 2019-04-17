import configparser
from datetime import datetime
import os

import pymongo
import tweepy

tweet_time_format = "%a %b %d %H:%M:%S +0000 %Y"

cnfg = configparser.ConfigParser()
cnfg.read("../data/twitdata.ini")

error_fn = "twitter_errors.txt"
if not os.path.isfile(error_fn):
    with open(error_fn, "w", encoding="utf-8") as out:
        out.write("response\tcode\tscraping_type\tid\n")

auth = tweepy.OAuthHandler(cnfg["Tweepy"]["consumer_key"], cnfg["Tweepy"]["consumer_secret"])
auth.set_access_token(cnfg["Tweepy"]["access_key"], cnfg["Tweepy"]["access_secret"])

twitter = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

mongo = pymongo.MongoClient(connect=False)
db = mongo.smm4h


def tweet_to_db(DB_COL, record):
    DB_COL.insert_one(record)


def clean_tweet(j):
    """Clean the tweet json from redundant fields. For example duplicate data in
    integer/string form and http/https forms of URLs.
    """

    # The following fields are deprecated: https://dev.twitter.com/overview/api/tweets
    del j["contributors"]
    del j["geo"]

    # these are redundant because we have `id_str` versions of them
    del j["id"]
    del j["in_reply_to_status_id"]
    del j["in_reply_to_user_id"]
    if j.get("quoted_status_id"):
        del j["quoted_status_id"]

    # user
    del j["user"]["id"]
    del j["user"]["profile_background_image_url"]  # profile_background_image_url_https
    del j["user"]["profile_image_url"]             # profile_image_url_https

    # entities
    if j["entities"].get("media"):
        for m in j["entities"]["media"]:
            del m["id"]         # id_str
            del m["media_url"]  # media_url_https
            if m.get("source_status_id"):
                del m["source_status_id"]  # source_status_id_str
    if j["entities"].get("user_mentions"):
        for mention in j["entities"]["user_mentions"]:
            del mention["id"]

    # retweet data
    if j.get("retweeted_status"):
        j["retweeted_status"] = clean_tweet(j["retweeted_status"])

    # truncated tweets, replace data with extended_tweet
    if j["truncated"]:
        if j.get("extended_tweet"):
            ext = j["extended_tweet"]
            if ext.get("full_text"):
                j["text"] = ext["full_text"]
            del j["extended_tweet"]
    del j["truncated"]

    return j


def filter_tweet(j):
    """Filter the tweet JSON from data we won't use."""
    if j.get("display_text_range"):
        del j["display_text_range"]
    if j.get("timestamp_ms"):
        del j["timestamp_ms"]

    def filter_entities(entities):
        if entities.get("media"):
            for m in entities["media"]:
                del m["url"]
                del m["display_url"]
                del m["expanded_url"]
                del m["indices"]
                del m["sizes"]

        if entities.get("hashtags"):
            for h in entities["hashtags"]:
                del h["indices"]

        if entities.get("symbols"):
            for s in entities["symbols"]:
                del s["indices"]

        if entities.get("urls"):
            for u in entities["urls"]:
                del u["indices"]
                del u["display_url"]
                del u["url"]

        if entities.get("user_mentions"):
            for m in entities["user_mentions"]:
                del m["indices"]
                del m["name"]

        return entities

    j["entities"] = filter_entities(j["entities"])
    if j.get("extended_entities"):
        j["extended_entities"] = filter_entities(j["extended_entities"])

    del j["user"]["contributors_enabled"]
    del j["user"]["profile_background_color"]
    del j["user"]["profile_background_image_url_https"]
    del j["user"]["profile_background_tile"]
    if j["user"].get("profile_banner_url"):
        del j["user"]["profile_banner_url"]
    del j["user"]["profile_image_url_https"]
    del j["user"]["profile_link_color"]
    del j["user"]["profile_sidebar_border_color"]
    del j["user"]["profile_sidebar_fill_color"]
    del j["user"]["profile_text_color"]
    del j["user"]["profile_use_background_image"]
    del j["user"]["default_profile"]
    del j["user"]["default_profile_image"]

    return j


def get_user_timeline(uid):
    try:
        for status in tweepy.Cursor(twitter.user_timeline, user_id=uid, tweet_mode="extended", exclude_replies=False, count=200).items():
            j = filter_tweet(clean_tweet(status._json))
            record = {
                "_id": j["id_str"],
                "tweet": j,
                "datetime": datetime.strptime(j["created_at"], tweet_time_format),
            }
            tweet_to_db(db.timelines, record)
    except tweepy.error.TweepError as e:
        with open(error_fn, "a") as error:
            error.write("{r}\t{c}\tuser_timeline\t{u}\n".format(u=uid, c=e.api_code, r=e.response))


def get_timelines(training=True):
    ufn = "../data/user_ids.txt" if training else "../data/task1/testDataST1_user_ids.txt"
    user_ids = set()
    with open(ufn, encoding="utf-8") as doc:
        for line in doc.readlines():
            uid = line.strip("\n")
            user_ids.add(int(uid))

    for uid in user_ids:
        get_user_timeline(uid)


def build_record(j, labels):
    # Add datetime version of created_at fields for querying on dates in DB.
    record = {
        "_id": j["id_str"],
        "tweet": j,
        "datetime": datetime.strptime(j["created_at"], tweet_time_format),
        "task_one": True if "one" in labels else False,
        "task_two": True if "two" in labels else False,
        "task_three": True if "three" in labels else False,
        "task_four": True if "four" in labels else False,
    }
    return record


def get_status(tid):
    try:
        status = twitter.get_status(tid)
        return status
    except tweepy.error.TweepError as e:
        with open(error_fn, "a") as error:
            error.write("{r}\t{c}\tget_status\t{u}\n".format(u=tid, c=e.api_code, r=e.response))
        return None


def get_tweets(training=True):
    fn = "../data/tweets_to_obtain.txt" if training else "../data/task1/testDataST1_participants.tsv"
    ufn = "../data/user_ids.txt" if training else "../data/task1/testDataST1_user_ids.txt"
    with open(fn) as doc:
        for line in doc.readlines():
            cols = line.strip("\n").split("\t")
            status = get_status(int(cols[0]))
            labels = cols[1].split(",")
            if status is not None:
                j = filter_tweet(clean_tweet(status._json))
                record = build_record(j, labels)
                tweet_to_db(db.tweets, record)
                with open(ufn, "a", encoding="utf-8") as out:
                    out.write("{uid}\n".format(uid=j["user"]["id_str"]))


def main():
    get_tweets(training=False)
    print("Started processing all user timelines")
    get_timelines(training=False)


if __name__ == '__main__':
    main()
