# Task 1

Description of task 1 files.


## Features

### Users
- [x] user.number_of_friends
- [x] user.log(number_of_friends)
- [x] user.number_of_followers
- [x] user.log(number_of_followers)
- [x] user.ratio_friends_followers
- [x] user.number_of_tweets
- [x] user.log(number_of_tweets)
- [x] user.number_of_positive_cases : the number of tweets that were deemed positive in the timeline
- [x] user.ratio_number_of_positive_cases
- [x] user.number_of_drugs
- [x] user.number_of_medical_terms
- [x] user.number_of_natural_products
- [x] user.number_of_cannabis

### Temporal
- [x] (categorical)temp.hour_of_day :
- [x] (categorical)temp.day_of_week :
- [x] (categorical)temp.season : wonder about geographical bias
- [ ] temp.burst_of_posting_window_(time) : how many posts there are in the whole user timeline before/after actual tweet.

### Textual
- [x] text.lenght_of_text
- [x] text.number_of_words
- [x] (vector)text.pos_vector : a vector representing the distribution of POS tagging.
- [x] (vector)text.sentiment_(sentiment-tool): a vector representing different sentiment tools
- [x] text.tfidf_1-grams
- [x] text.tfidf_2-grams
- [x] text.tfidf_3-grams : (if possbile)

### Dictionary based
- [x] dict.number_of_drugs
- [x] dict.number_of_medical_terms
- [x] dict.number_of_natural_products : includes Cannabis
- [x] dict.tfidf_1-grams
- [x] dict.tfidf_2-grams
- [x] dict.tfids_3-grams : (if possible)


A feature example. Note that fields with NaN are omitted.

```json
> db.task_1_train_features.findOne()
{
  "_id" : ObjectId("5cb629fcd9b548b6545955c5"),
  "y" : 0,
  "user_number_friends" : 145,
  "user_log(number_friends)" : 4.976734,
  "user_number_followers" : 232,
  "user_log(number_followers)" : 5.446737,
  "user_ratio_friends_followers" : 0.625,
  "user_number_tweets" : 6059,
  "user_log(number_tweets)" : 8.7093,
  "temp_hour_of_day" : 4,
  "temp_day_of_week" : "saturday",
  "temp_season" : "summer",
  "timeline_length_text" : 10047,
  "timeline_number_words" : 1649,
  "timeline_number_(MedicalTerms)" : 22,
  "timeline_number_(Drugs)" : 1,
  "post_length_text" : 75,
  "post_number_words" : 14,
  "post_number_(Drugs)" : 1,
  "post_number_(MedicalTerms)" : 1,
  "post_tfidf_(enbrel)" : 0.721502,
  "post_tfidf_(today)" : 0.692412,
  "post_tfidf_parent_(etanercept)" : 0.707107,
  "post_tfidf_parent_(water)" : 0.707107
}
```

## Command to export features to json

```bash
mongoexport -vv -d smm4h -c task_1_train_features -o task_1_train_features.json
```
