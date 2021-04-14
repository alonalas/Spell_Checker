import re
from collections import Counter
from nltk import ngrams
import math
import random

class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    """Initializing a spell checker object with a language model as an
    instance  variable.

    Args:
    lm: a language model object. Defaults to None
    """
    def __init__(self, lm = None):
        self.lm = lm

    """Adds the specified language model as an instance variable.
    (Replaces an older LM dictionary if set)

    Args:
    lm: a Spell_Checker.Language_Model object
    """
    def add_language_model(self, lm):
        self.lm = lm

    """ Adds the speficied dictionary of error tables as an instance variable.
    (Replaces an older value disctionary if set)

    Args:
    error_tables (dict): a dictionary of error tables in the format
    returned by  learn_error_tables()
    """
    def add_error_tables(self, error_tables):
        self.error_table = error_tables

    """Returns the log-likelihod of the specified text given the language
    model in use. Smoothing should be applied on texts containing OOV words

    Args:
    text (str): Text to evaluate.

    Returns:
    Float. The float should reflect the (log) probability.
    """
    def evaluate(self, text):
        return 0

    """ Returns the most probable fix for the specified text. Use a simple
        noisy channel model is the number of tokens in the specified text is
        smaller than the length (n) of the language model.

        Args:
            text (str): the text to spell check.
            alpha (float): the probability of keeping a lexical word as is.

        Return:
            A modified string (or a copy of the original if no corrections are made.)
    """
    def spell_check(self, text, alpha):
        return 0

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns a model from a given text.
            It supoprts language generation and the evaluation of a given string.
            The class can be applied on both word level and character level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
                n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
                chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                              Defaults to False
            """
            self.n = n
            self.model_dict = None  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            self.vocabulary = None
            self.n_minus_one_dict = None

        """populates the instance variable model_dict.
        Args:
            text (str): the text to construct the model from.
        """
        def build_model(self, text):
            my_ngrams = ngrams(re.findall(r'\w+', text.lower()), self.n)
            self.model_dict = dict(Counter(my_ngrams))
            for old_key in list(self.model_dict.keys()):
                self.model_dict[" ".join(old_key)] = self.model_dict[old_key]
                del self.model_dict[old_key]
            self.n_minus_one_dict = dict(Counter(ngrams(re.findall(r'\w+', text.lower()), self.n-1)))
            for old_key in list(self.n_minus_one_dict.keys()):
                self.n_minus_one_dict[" ".join(old_key)] = self.n_minus_one_dict[old_key]
                del self.n_minus_one_dict[old_key]
            self.vocabulary = dict(Counter(re.findall(r'\w+', text.lower())))

        def get_model_dictionary(self):
            return self.model_dict

        def get_model_window_size(self):
            return self.n

        def create_context_randomly(self, n):
            random_ngram = random.choices(self.model_dict.keys(), weights=list(self.model_dict.values()), k=1)[0]
            context = random_ngram
            if self.n < n:
                for i in range (0,n-self.n):
                    end = random_ngram.split(' ', 1)[1]
                    sub_dict = {k: v for k, v in self.model_dict.items() if k.startswith(end + " ")}
                    addition = random.choices(sub_dict.keys(), weights=list(sub_dict.values()),
                                                       k=1)[0].split()[-1]
                    context = context + " " + addition
                    random_ngram = end + " "+ addition
                return context

            elif self.n == n: #one ngram is enough
                return random.choices(self.model_dict.keys(), weights=list(self.model_dict.values()), k=1)[0]
            else: #ngrams are greater than n, choose a random ngram and substring it to size n
                context = random.choices(self.model_dict.keys(), weights=list(self.model_dict.values()), k=1)[0]
                words = context.split()
                prefix = words[0]
                for i in range(1, n):
                    prefix = prefix + " " + words[i]
                return prefix

        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted. If the length of the specified context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified context.

        Args:
        context (str): a seed context to start the generated string from. Defaults to None
        n (int): the length of the string to be generated.

        Return:
        String. The generated text.

        """
        def generate(self, context=None, n=20):
            if context == None:
                context = self.create_context_randomly(self,n)

            elif len(context) >= n:
                words = context.split()
                prefix = words[0]
                for i in range(1, n):
                    prefix = prefix + " " + words[i]
                return prefix

            else: #regular case, a given context in size smaller than n-> complete the sentence
                return 0

        """Returns the smoothed (Laplace) probability of the specified ngram.
        Args:
        ngram (str): the ngram to have it's probability smoothed
        Returns:
        float. The smoothed probability.
        """
        def smooth(self, ngram):
            ngram_minus_one = ngram.rsplit(' ', 1)[0]
            mone = self.model_dict[ngram] + 1
            mechane = self.n_minus_one_dict[ngram_minus_one] + len(self.n_minus_one_dict)
            return mone/mechane

        """Returns the log-likelihood of the specified text to be a product of the model.
        Laplace smoothing should be applied if necessary.
        Args:
        text (str): Text to evaluate.
        Returns:
        Float. The float should reflect the (log) probability.
        """

        def evaluate(self, text):
            text_ngrams = ngrams(re.findall(r'\w+', text.lower()), self.n)
            ngrams_probabilities = {}
            for ngram in text_ngrams:
                ngram_minus_one = ngram.rsplit(' ', 1)[0]
                if (ngram not in self.model_dict) or (ngram_minus_one not in self.n_minus_one_dict):
                    probability = self.smooth(ngram)
                else:
                    probability = self.model_dict[ngram] / self.n_minus_one_dict[ngram_minus_one]
                ngrams_probabilities[ngram] = probability
            multiply = 1
            for ngram in ngrams_probabilities:
                multiply = multiply * ngrams_probabilities[ngram]
            return math.log10(multiply)


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """


def who_am_i():
    return {'name': 'Alona Lasry', 'id': '205567944', 'email': 'alonalas@post.bgu.ac.il'}