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
- [x] temp.hour_of_day
- [x] temp.day_of_week
- [x] temp.season : wonder about geographical bias
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
db.task_1_train_features.findOne()
{
  "_id" : ObjectId("5ca79c8fd9b5480b525ba812"),
  "y" : 0,
  "user_number_of_friends" : 145,
  "user_log(number_of_friends)" : 4.976733742420574,
  "user_number_of_followers" : 232,
  "user_log(number_of_followers)" : 5.44673737166631,
  "user_ratio_friends_followers" : 0.625,
  "user_number_of_tweets" : 6059,
  "user_log(number_of_tweets)" : 8.70930004894499,
  "temp_hour_of_day" : 4,
  "temp_day_of_week" : "saturday",
  "temp_season" : "summer",
  "post_length_text" : 75,
  "post_number_words" : 14,
  "post_number_(NOUN+VERB+ADJ)" : 0,
  "post_number_(Drugs)" : 1,
  "post_number_(MedicalTerms)" : 1,
  "post_number_(NaturalProducts)" : 0,
  "text_number_of_(VERB)" : 0,
  "text_number_of_(NOUN)" : 0,
  "text_number_of_(PRON)" : 0,
  "text_number_of_(ADJ)" : 0,
  "text_number_of_(ADV)" : 0,
  "text_number_of_(ADP)" : 0,
  "text_number_of_(CONJ)" : 0,
  "text_number_of_(DET)" : 0,
  "text_number_of_(NUM)" : 0,
  "text_number_of_(PRT)" : 0,
  "text_number_of_(X)" : 0,
  "text_number_of_(pct)" : 0,
  "post_tfidf(enbrel)" : 0.721501973006662,
  "post_tfidf(today)" : 0.6924123792563893,
  "post_tfidf-parent(etanercept)" : 0.7071067811865476,
  "post_tfidf-parent(water)" : 0.7071067811865476
}
```
