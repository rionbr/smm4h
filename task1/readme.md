# Task 1

Description of task 1 files.


## Features

### Users
- us_number_of_friends
- us_log_number_of_friends
- us_number_of_followers
- us_log_number_of_followers
- us_ratio_friends_followers
- us_number_of_tweets
- us_log_number_of_tweets
- us_number_of_positive_cases : the number of tweets that were deemed positive in the timeline
- us_ratio_number_of_positive_cases
- us_number_of_drugs
- us_number_of_medical_terms
- us_number_of_natural_products
- us_number_of_cannabis

### Temporal
- tm_hour_of_day
- tm_day_of_week
- tm_season : wonder about geographical bias
- tm_burst_of_posting_window_(time) : how many posts there are in the whole user timeline before/after actual tweet.

### Textual
- tx_lenght_of_text
- tx_number_of_words
- (vector)tx_pos_vector : a vector representing the distribution of POS tagging.
- (vector)tx_sentiment_(sentiment-tool): a vector representing different sentiment tools
- tx_tfidf_1-grams
- tx_tfidf_2-grams
- tx_tfidf_3-grams : (if possbile)

### Dictionary based
- dc_number_of_drugs
- dc_number_of_medical_terms
- dc_number_of_natural_products
- dc_number_of_cannabis
- dc_tfidf_1-grams
- dc_tfidf_2-grams
- dc_tfids_3-grams : (if possible)
