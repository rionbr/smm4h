# Sentiment scoring for tweets

Using Sentiment the texts of tweets can be scored with several sentiment tools.


## Dependencies

To install the python dependencies, run the following commands:

``` shell
pip install nltk
pip install afinn
```

Next, run `setup.py` to load all NLTK libraries used by the tool.

``` shell
python sentiment.py
```

After this, the tool is ready to use.

## Usage

``` python
from sentiment import Sentiment

text = "This is a test, let's see if we get a good result!"
mysent = Sentiment()

sentiment_scores = mysent.calculate_average_score(text)
```
