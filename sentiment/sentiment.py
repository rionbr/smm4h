import os

import pandas as pd

from afinn import Afinn
import nltk

from .vaderSentiment import SentimentIntensityAnalyzer

_FOLDER = os.path.dirname(__file__)
_SENTIMENT_SCORES = ["VADER_Valence", "VADER_Arousal", "VADER_Dominance", "VADER_OpinionFinder", "VADER", "Afinn_Valence", "OpinionFinder",
                     "ANEW_Valence", "ANEW_Arousal", "ANEW_Dominance", "GPOMS_composed/anxious", "GPOMS_agreeable/hostile",
                     "GPOMS_elated/depressed", "GPOMS_confident/unsure", "GPOMS_clearheaded/confused", "GPOMS_energetic/tired"]


class Sentiment(object):

    def __init__(self):
        self.afinn = self.load_afinn()
        self.anew = self.load_anew(sort="Mean")
        self.anew_std = self.load_anew(sort="SD")
        self.opinion_finder = self.load_opinion_finder()
        self.vader_scorer = self.load_vader()

    def load_afinn(self):
        """Loads the Afinn [1] Valence lexicon and sentiment scores.

        Returns
        -------
        Afinn Valence scores : pandas.DataFrame
            DataFrame containing Afinn Valence lexicon and scores

        [1] Finn Årup Nielsen, "A new ANEW: evaluation of a word list for sentiment analysis in microblogs",
            Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages.
            Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)
        """
        afinn = pd.DataFrame.from_dict(Afinn(language="en", emoticons=True)._dict, orient="index", columns=["Afinn_Valence"])
        afinn.index.rename("Word", inplace=True)
        return afinn

    def load_vader(self):
        """Loads the VADER [1] lexicon and sentiment analyzer.

        Returns
        -------
        VADER sentiment analyzers : dictionary
            dictionary with several versions of the VADER sentiment analyzer based on different lexicons,
            i.e., original, ANEW Valence, ANEW Dominance, ANEW Arousal, and OpinionFinder

        [1] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
            Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
        """
        VADER = pd.read_csv('{fldr}/data/vader_lexicon.txt'.format(fldr=_FOLDER), sep='\t', header=None)
        VADER = VADER[[0, 1]]
        VADER.columns = ["Word", "VADER"]
        VADER.set_index("Word", inplace=True)
        self.vader = VADER
        return {
            "VADER_Valence": SentimentIntensityAnalyzer(lexicon_file='{fldr}/data/Anew_valence.txt'.format(fldr=_FOLDER)),
            "VADER_Arousal": SentimentIntensityAnalyzer(lexicon_file='{fldr}/data/Anew_arousal.txt'.format(fldr=_FOLDER)),
            "VADER_Dominance": SentimentIntensityAnalyzer(lexicon_file='{fldr}/data/Anew_dominance.txt'.format(fldr=_FOLDER)),
            "VADER_OpinionFinder": SentimentIntensityAnalyzer(lexicon_file='{fldr}/data/OpFi-Sent.txt'.format(fldr=_FOLDER)),
            "VADER": SentimentIntensityAnalyzer(),
        }

    def load_anew(self, sort="Mean"):
        """Loads the CRR ANEW [1] lexicon and sentiment scores.

        Parameters
        ----------
        sort : string
            type of ANEW scores that have to be returned (Mean for mean ANEW values, SD for standard deviation of ANEW values)

        Returns
        -------
        ANEW scores : pandas.DataFrame
            DataFrame containing ANEW lexicon and scores

        [1] Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
            Behavior Research Methods, 45, 1191-1207. (http://crr.ugent.be/archives/1003)
        """
        if sort not in ["Mean", "SD"]:
            raise(NotImplementedError)

        anew = pd.read_csv("{fldr}/data/Ratings_Warriner_et_al.csv".format(fldr=_FOLDER), index_col="Word")
        cols = ['{t}.{s}.Sum'.format(t=t, s=sort) for t in ["V", "A", "D"]]
        anew = anew[cols]
        anew.columns = ["Valence", "Arousal", "Dominance"]
        return anew

    def load_opinion_finder(self):
        """Loads the OpinionFinder [1] lexicon and sentiment scores.

        Returns
        -------
        OpinionFinder scores : pandas.DataFrame
            DataFrame containing OpinionFinder lexicon and scores

        [1] Wilson, T. & Hoffmann, P. & Somasundaran, S. & Kessler, J. & Wiebe, J. & Choi, Y. & Cardie, C. & Riloff, E. & Patwardhan, S. (2005).
            OpinionFinder: A System for Subjectivity Analysis. HLT/EMNLP 2005: 2005. 10.3115/1225733.1225751. (http://mpqa.cs.pitt.edu/opinionfinder/)
        """
        OpFi = pd.read_csv("{fldr}/data/OpFi-Sent.txt".format(fldr=_FOLDER), sep="\t", header=None)
        OpFi.columns = ["Word", "OpinionFinder"]
        OpFi.set_index("Word", inplace=True)
        return OpFi

    def calculate_average_score(self, text):
        """Scores a text using several sentiment scores.

        Parameters
        ----------
        text : string
            text that has to be processed

        Returns
        -------
        sentiment_results : dictionary of scores
            Average sentiment score for all scored sentiments based on text
        """
        words = tokenize_text(text)

        sentiment_results = self.score_vader(text)
        afinn_results = self.score_afinn(words)
        if afinn_results:
            sentiment_results["Afinn_Valence"] = afinn_results["Afinn_Valence"]
        else:
            sentiment_results["Afinn_Valence"] = pd.np.nan

        opinion_finder_results = self.score_opinion_finder(words)
        if opinion_finder_results:
            sentiment_results["OpinionFinder"] = opinion_finder_results["OpinionFinder"]
        else:
            sentiment_results["OpinionFinder"] = pd.np.nan

        anew_results = self.score_anew(words)
        if anew_results:
            for sent in anew_results:
                sentiment_results["ANEW_" + sent] = anew_results[sent]
        else:
            for sent in anew_results:
                sentiment_results["ANEW_" + sent] = pd.np.nan

        return sentiment_results

    def score_vader(self, text, return_all=False):
        """Scores a text using VADER [1].

        Parameters
        ----------
        text : str
            text that has to be processed
        return_all: boolean (optional)
            sets wether to output all word scores (True) or just the mean (False).
            Only uses the complete VADER sentiment scorer if set to True, if set to False
            it will perform a dictionary matching based on a tokenized text.

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            VADER sentiment scores for text

        [1] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
            Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
        """
        if return_all:
            self.score_dataframe(self.vader, tokenize_text(text), return_all)
        else:
            results = {}
            for sent in self.vader_scorer:
                results[sent] = self.vader_scorer[sent].polarity_scores(text)["compound"]
            return results

    def score_afinn(self, words, return_all=False):
        """Scores a list of words using Afinn [1] Valence.

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional)
            sets wether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            Afinn Valence scores for words

        [1] Finn Årup Nielsen, "A new ANEW: evaluation of a word list for sentiment analysis in microblogs",
            Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages.
            Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)
        """
        return self.score_dataframe(self.afinn, words, return_all)

    def score_opinion_finder(self, words, return_all=False):
        """Scores a list of words using OpinionFinder [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pd.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            OpinionFinder scores for words

        [1] Wilson, T. & Hoffmann, P. & Somasundaran, S. & Kessler, J. & Wiebe, J. & Choi, Y. & Cardie, C. & Riloff, E. & Patwardhan, S. (2005).
            OpinionFinder: A System for Subjectivity Analysis. HLT/EMNLP 2005: 2005. 10.3115/1225733.1225751. (http://mpqa.cs.pitt.edu/opinionfinder/)
        """
        return self.score_dataframe(self.opinion_finder, words, return_all)

    def score_anew(self, words, return_all=False):
        """Scores a list of words using ANEW.

        Parameters
        ----------
        words : iterable of str
            words to be processed
        return_all: boolean (optional, default False)
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            ANEW scores for words
        """
        return self.score_dataframe(self.anew, words, return_all)

    def score_anew_std(self, words):
        """Scores a list of words using CRR ANEW [1].

        Parameters
        ----------
        words : iterable of str
            words to be processed

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            ANEW SD values for words

        [1] Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
            Behavior Research Methods, 45, 1191-1207. (http://crr.ugent.be/archives/1003)
        """
        return self.score_dataframe(self.anew_std, words, False)

    def score_dataframe(self, df, words, return_all):
        """Scores a list of words using CRR ANEW [1].

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe of words (index) vs scores (columns)
        words : iterable of str
            words to be processed
        return_all: boolean
            sets whether to output all word scores (True) or just the mean (False)

        Returns
        -------
        scores : pandas.DataFrame of scores (return_all=True) or dictionary of average score (return_all=False)
            Sentiment scores for words based on input DataFrame

        [1] Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas.
            Behavior Research Methods, 45, 1191-1207. (http://crr.ugent.be/archives/1003)
        """
        matches = []
        for word in words:
            if word in df.index:
                matches.append(word)

        scores = df.loc[matches, :]
        if not return_all:
            return scores.mean(axis=0).to_dict()
        else:
            return scores


def tokenize_text(text):
    """Tokenizes a given text

    Parameters
    ----------
    text : string
        The text that has to be tokenized

    Returns
    -------
    words : list of tokens
    """
    words = nltk.word_tokenize(text)
    return words
