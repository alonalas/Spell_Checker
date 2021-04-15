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

    def __init__(self, lm=None):
        self.lm = lm

    def add_language_model(self, lm):
        self.lm = lm

    def add_error_tables(self, error_tables):
        self.error_table = error_tables

    def evaluate(self, text):
        return self.lm.evaluate(text)

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
        splitted_text = text.split()
        for i in range(0, len(splitted_text)):
            if splitted_text[i] not in self.lm.vocabulary:
                if splitted_text[i] in self.error_table:
                    splitted_text[i] = self.error_table[splitted_text[i]]
                else:
                    candidates = self.candidates(splitted_text[i])
                    max_candidate = self.calculate_max_candidate(candidates, type([candidates][0] is tuple))
                    splitted_text[i] = max_candidate
                return " ".join(splitted_text)
        # we only achieved here if the text does not include a word that doesnt exist (assumption: a text includes only one error)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return self.known([word], None) or self.known(self.edits1(word), "edit") or self.known(self.edits2(word),
                                                                                               "edit") or [word]

    def known(self, words, method):
        "The subset of `words` that appear in the dictionary of WORDS."
        if method == "edit":
            known = []
            only_words = [a_tuple[0] for a_tuple in words]
            for w in only_words:
                if w in self.lm.vocabulary:
                    for a_tuple in words:
                        if a_tuple[0] == w:
                            known.append((w, a_tuple[1]))
            return set(known)

        else:  # none
            return set(w for w in words if w in self.lm.vocabulary)

    # probability of word
    def P(self, candidate, corrected):
        if not corrected:
            return self.lm.vocabulary[candidate] / sum(self.lm.vocabulary.values())
        else:
            correction = 0
            self.error_table[candidate[1][]]

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [(L + R[1:], "deletion") for L, R in splits if R]
        transposes = [(L + R[1] + R[0] + R[2:], "transposition") for L, R in splits if len(R) > 1]
        replaces = [(L + c + R[1:], "substitution") for L, R in splits if R for c in letters]
        inserts = [(L + c + R, "insertion") for L, R in splits for c in letters]

        the_set = set(deletes + transposes + replaces + inserts)
        return the_set

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def calculate_max_candidate(self, candidates, corrected):
        if corrected:  # candidates are tuples-> multiply prior with p(x|y)
            return max(candidates,key=self.P, True)
        else:
            return max(candidates, key=self.P, False)

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:

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

        def build_model(self, text):
            my_ngrams = ngrams(re.findall(r'\w+', text.lower()), self.n)
            self.model_dict = dict(Counter(my_ngrams))
            for old_key in list(self.model_dict.keys()):
                self.model_dict[" ".join(old_key)] = self.model_dict[old_key]
                del self.model_dict[old_key]
            self.n_minus_one_dict = dict(Counter(ngrams(re.findall(r'\w+', text.lower()), self.n - 1)))
            for old_key in list(self.n_minus_one_dict.keys()):
                self.n_minus_one_dict[" ".join(old_key)] = self.n_minus_one_dict[old_key]
                del self.n_minus_one_dict[old_key]
            self.vocabulary = dict(Counter(re.findall(r'\w+', text.lower())))

        def get_model_dictionary(self):
            return self.model_dict

        def get_model_window_size(self):
            return self.n

        def complete_context(self, n, context):
            words = context.split()
            current_ngram = context
            if len(words) > self.n:
                last_n_words = words[len(words) - self.n]
                for j in range(len(words) - self.n + 1, len(words)):
                    last_n_words = last_n_words + " " + words[j]
                current_ngram = last_n_words

            for i in range(0, n - self.n):
                end = current_ngram.split(' ', 1)[1]
                sub_dict = {k: v for k, v in self.model_dict.items() if k.startswith(end + " ")}
                shortened_end = end.split(' ', 1)[1]
                while len(sub_dict) == 0:
                    sub_dict = {k: v for k, v in self.model_dict.items() if k.startswith(shortened_end + " ")}
                    if len(shortened_end.split()) > 1:
                        shortened_end = shortened_end.split(' ', 1)[1]
                    elif len(shortened_end.split()) == 1 and shortened_end not in self.vocabulary.keys():
                        return context
                    elif len(shortened_end.split()) == 1 and shortened_end in self.vocabulary.keys():
                        break

                addition = random.choices(list(sub_dict.keys()), weights=list(sub_dict.values()),
                                          k=1)[0].split()[-1]
                context = context + " " + addition
                current_ngram = end + " " + addition
            return context

        def create_context_randomly(self, n):
            if self.n < n:
                first_ngram = random.choices(list(self.model_dict.keys()), weights=list(self.model_dict.values()), k=1)[
                    0]
                return self.complete_context(n, first_ngram)
            elif self.n == n:  # one ngram is enough
                return random.choices(self.model_dict.keys(), weights=list(self.model_dict.values()), k=1)[0]
            else:  # ngrams are greater than n, choose a random ngram and substring it to size n
                context = random.choices(self.model_dict.keys(), weights=list(self.model_dict.values()), k=1)[0]
                return self.generate_prefix_sized_n(n, context)

        def generate_prefix_sized_n(self, n, context):
            words = context.split()
            prefix = words[0]
            for i in range(1, n):
                prefix = prefix + " " + words[i]
            return prefix

        def generate(self, context=None, n=20):
            if context == None:
                return self.create_context_randomly(n)
            else:
                words = context.split()
                if len(words) >= n:
                    return self.generate_prefix_sized_n(n, context)
                else:
                    if self.n > len(words):  # not enough words in the first ngram
                        sub_dict = {k: v for k, v in self.model_dict.items() if k.startswith(context + " ")}
                        if len(sub_dict) != 0:
                            return self.complete_context(n, random.choices(list(sub_dict.keys()),
                                                                           weights=list(sub_dict.values()), k=1)[0])
                        else:
                            if len(words) == 1:
                                return context
                            else:
                                shortened_end = context.split(' ', 1)[1]
                                while len(sub_dict) == 0:
                                    sub_dict = {k: v for k, v in self.model_dict.items() if
                                                k.startswith(shortened_end + " ")}
                                    if len(shortened_end.split()) > 1:
                                        shortened_end = shortened_end.split(' ', 1)[1]
                                    elif len(
                                            shortened_end.split()) == 1 and shortened_end not in self.vocabulary.keys():
                                        return context
                                    elif len(shortened_end.split()) == 1 and shortened_end in self.vocabulary.keys():
                                        break

                                addition = random.choices(list(sub_dict.keys()), weights=list(sub_dict.values()),
                                                          k=1)[0].split()[-1]
                                random_ngram_completion = context + " " + addition
                                return self.complete_context(n, random_ngram_completion)
                    else:  # regular case, a given context in size smaller than n-> complete the sentence
                        return self.complete_context(n, context)

        def smooth(self, ngram):
            ngram_minus_one = ngram.rsplit(' ', 1)[0]
            if ngram not in self.model_dict:
                self.model_dict[ngram] = 1
            if ngram_minus_one not in self.n_minus_one_dict:
                self.n_minus_one_dict[ngram_minus_one] = 1

            mone = self.model_dict[ngram]
            mechane = self.n_minus_one_dict[ngram_minus_one] + len(self.n_minus_one_dict)
            return mone / mechane

        def evaluate(self, text):
            text_ngrams = ngrams(re.findall(r'\w+', text.lower()), self.n)
            ngrams_probabilities = {}
            for ngram in text_ngrams:
                ngram = " ".join(ngram)
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
