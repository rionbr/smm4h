FORMAT OF THE DATA:

TWEETID <-tab-> USERID <-tab-> CATEGORY (1=ADR; 0=NON-ADR)

Total number of files present: 3
——————————
CITATION

Sarker A et al. Data and systems for medication-related text classification and concept normalization from Twitter: Insights from the Social Media Mining for Health (SMM4H)-2017 shared task. Journal of the American Medical Informatics Association. 2018. doi: 10.1093/jamia/ocy114.

———————————
DOWNLOAD INSTRUCTIONS

Please note that by downloading the Twitter data you agree to follow the Twitter terms of service (https://twitter.com/tos).

You MUST NOT re-distribute the tweets, the annotations or the corpus obtained, as this violates the Terms of Use.

The download_tweets.py scripts downloads the tweet texts (if still available) using the user and tweet IDs. To download:

Requirements: 

python 2.7.* (beautifulsoup4, json)

Command: python download_tweets.py input_filename > output_filename

Sample: python download_tweets.py training_set_1_ids.txt > training_set_1_tweets.txt

———————————
Contact: abeed@pennmedicine.upenn.edu
