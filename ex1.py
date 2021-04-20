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
        """Initializing a spell checker object with a language model as an
        instance  variable
        instance variable 'calculated_candidates' is a dictionary that stores the probability
        for a candidate word to be the right correction of the parameter text to check.
        it is used every time when calculating P of a candidate,
        instead calculating it over again if it already have been calculated in the past
        :param lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.calculated_candidates = {}

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
        (Replaces an older LM dictionary if set)
        :param lm: a Spell_Checker.Language_Model object
        """
        self.lm = lm

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
        (Replaces an older value disctionary if set)
        :param error_tables: a dictionary of error tables in the format
        """
        self.error_table = error_tables

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
        model in use. Smoothing should be applied on texts containing OOV words

        :param text (str): Text to evaluate.
        :return Float. The float should reflect the (log) probability.
        """

        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
        noisy channel model if the number of tokens in the specified text is
        smaller than the length (n) of the language model.

        :param text (str): the text to spell check.
        :param alpha (float): the probability of keeping a lexical word as is.
        :return A modified string (or a copy of the original if no corrections are made.)
        """
        self.text_to_check = text
        splitted_text = text.split()

        for i in range(0, len(splitted_text)):
            if splitted_text[i] not in self.lm.vocabulary:
                self.unreal_word_index = i
                candidates = self.candidates(splitted_text[i])
                max_candidate = max(candidates, key=self.P)
                splitted_text[i] = max_candidate[0]
                return " ".join(splitted_text)

        # we only achieved here if the text does not include a word that doesnt exist (assumption: a text includes
        # only one error)
        self.alpha = alpha  # the probability will be calculated as 1-alpha => The probability that the user typed a word he did not mean is low

        #original_sentence_prob = self.calc_probability_depends_on_context()*alpha
        original_sentence_prob = {}
        for i in range(0, len(splitted_text)):
            original_sentence_prob[i] = self.calc_probability_depends_on_context(splitted_text[i])*0.95

        Ps = {}
        for wrong_word_idx in range(0, len(splitted_text)):
            self.unreal_word_index = wrong_word_idx
            list_candidates = self.candidates(splitted_text[wrong_word_idx])
            for candidate in list_candidates:
                Ps[self.P(candidate)] = (candidate[0], wrong_word_idx)

        max_prob = max(Ps.keys())
        if original_sentence_prob[Ps[max_prob][1]] > max_prob:
            return text
        else:
            max_candidate = Ps[max_prob]
            splitted_text[max_candidate[1]] = max_candidate[0]
            return " ".join(splitted_text)

    def candidates(self, word):
        """
        The method gets a word that is suspicious to be misspelled as input, and returns a set of candidate words
        where each word is a possible correction of the input word
        :param word: suspicious misspelled word
        :return: set of candidate words where each word is a possible correction of the input word
        """
        return self.known(self.edits1(word)) | self.known(self.edits2(word))

    def known(self, words):
        """
        The method gets a set of candidate words as parameter, The words can be familiar in the language or not.
        the function checks for each word if the word is familiar by checking if it exists in the language model vocabulary,
        if the vocabulary has it- than the word is appended to the result set.
        :param words: a set of candidate words
        :return: a subset of the candidate words that are known by the language vocabulary
        """
        known = []
        for a_tuple in words:
            word = a_tuple[0]
            if word in self.lm.vocabulary:
                if len(a_tuple) == 2:  # edit1
                    known.append((word, a_tuple[1]))
                else:  # edit2
                    known.append((word, a_tuple[1], a_tuple[2]))
        the_set = set(known)
        return the_set

    def P(self, candidate):
        """
        The method calculates the probability of a word, P(word) based on the noisy channel
        :param candidate: The word for which the probability should be calculated
        :return: The probability based on the noisy channel
        """
        if len(candidate) == 2:  # edit1 returned word
            return self.calculate_P_once(candidate)
        else:  # edit2 returned word
            return self.calculate_P_once((candidate[1][0], candidate[1][1])) * self.calculate_P_once((candidate[0], candidate[2]))

    def calculate_P_once(self, candidate):
        """
        Auxiliary function that calculates the noisy channel probability in case it can be calculated
        (the given text is in length greater than the model's n), else the calculation based on the
        simple noisy channel (with priors)
        :param candidate:
        :return: the probability
        """
        keys = candidate[1].split()
        if keys[1] in self.error_table[keys[0]]:
            count_mone = self.error_table[keys[0]][keys[1]]
            if count_mone == 0:
                return 0
            else:
                N = len(self.lm.vocabulary)
                try:
                    mechane = self.calculate_normalization_for_noisy_channel(keys[0], keys[1])
                except ZeroDivisionError:
                    return 0
                if candidate[0] in self.lm.vocabulary:
                    noisy_channel = count_mone / mechane
                    if len(self.text_to_check.split()) < self.lm.n:  # calculate with prior
                        prior = self.lm.vocabulary[candidate[0]] / N
                        return noisy_channel * prior
                    else:
                        if candidate[0] not in self.calculated_candidates:
                            context_evaluate = self.calc_probability_depends_on_context(candidate[0])
                            self.calculated_candidates[candidate[0]] = context_evaluate
                            return noisy_channel * context_evaluate
                        else:
                            return noisy_channel * self.calculated_candidates[candidate[0]]
                else:
                    return 0
        else:
            return 0

    def calc_probability_depends_on_context(self, candidate):
        """
        the function gets the candidate word for the correction of the text, and uses the method evaluate for calculating
        the probability that the given text is a product of the language model. the calculation based on the ngrams
        that includes the candidate in the given context.
        i.e:
        text = "the acress ate the apple",
        candidate = "actress"
        ngrams: "the actress ate", "actress ate the" (assume that the model's n = 3)
        :param candidate: the candidate word correction
        :return: the probability of the conext corrected by the candidate word
        """
        splitted_text = self.text_to_check.split()
        splitted_text[self.unreal_word_index] = candidate
        text = " ".join(splitted_text)
        my_ngrams = ngrams(re.findall(r'\w+', text.lower()), self.lm.n)
        multiply = 1
        for ngram in my_ngrams:
            if candidate in " ".join(ngram):
                multiply = multiply * math.pow(10, self.evaluate(" ".join(ngram)))
        return multiply

    def calculate_normalization_for_noisy_channel(self, error_type, the_correction):
        """
        The function calculates the denominator for the noisy channel calculation, depends on each edition rule
        :param error_type: the edition rule, String-> deletion/insertion/substitution/transposition
        :param the_correction: the misspelled charecters. i.e ab->ba
        :return:
        """
        count = 0
        if error_type == "deletion" or error_type == "transposition":
            if "#" in the_correction:
                the_correction = the_correction[1]
            for word in self.lm.vocabulary:
                if the_correction in word:
                    count += 1
        elif error_type == "insertion":
            if "#" in the_correction:
                the_correction = the_correction[1]
            for word in self.lm.vocabulary:
                if the_correction[0] in word:
                    count += 1
        else:  # error_type == "substitution":
            for word in self.lm.vocabulary:
                if the_correction[1] in word:
                    count += 1
        return count

    def edits1(self, word):
        """
        a simple edit to a word is a deletion (remove one letter), a transposition (swap two adjacent letters),
        a replacement (change one letter to another), or an insertion (add a letter)
        :param word: the word to edit
        :return: a set of all the edited strings that can be made with one simple edit, each element of the set is a tuple
        (string, string) where the first entry is the edited word and the second entry is the deition rule
        (deletion/ insertion...) and the edition (i.e ab->ba)
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [(L + c + R, "deletion #" + c) if L[0:1] == "" else (L + c + R, "deletion " + L[-1] + c) for L, R
                   in splits for c in letters]
        inserts = [(L + R[1:], "insertion #" + R[0:1]) if L[len(L) - 1:len(L)] == "" else (
            L + R[1:], "insertion " + L[len(L) - 1:len(L)] + R[0:1]) for L, R in splits if R]
        transposes = [(L + R[1] + R[0] + R[2:], "transposition " + R[1] + R[0] ) for L, R in splits if len(R) > 1]
        replaces = [(L + c + R[1:], "substitution " + R[0:1] + c) for L, R in splits if R for c in letters]

        the_set = set(deletes + transposes + replaces + inserts)
        return the_set

    def edits2(self, word):
        """
        Uses edits1 twice for generating a new edited string that require two simple edits.
        :param word: the word to edit
        :return: a set of all the edited strings that can be made with one simple edit, each element of the set is a triple
        (string, tuple, string)
        where the first entry is the edited word after the 2nd edition,
        the second entry is the output of edits1 for the first edition
        and the second entry is the deition rule (deletion/ insertion...) and the edition (i.e ab->ba) for the 2nd edition
        """
        set_e2 = []
        for e1 in self.edits1(word):
            for e2 in self.edits1(e1[0]):
                if not e2[0] == word:
                    set_e2.append((e2[0], e1, e2[1]))
        return set(set_e2)

    #####################################################################
    #                   Inner class                                     #
    #####################################################################

    class Language_Model:
        """The class implements a Markov Language Model that learns amodel from a given text.
        It supoprts language generation and the evaluation of a given string.
        The class can be applied on both word level and caracter level.
        """

        def __init__(self, n=3, chars=False):
            """Initializing a language model object.
            Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
            Defaults to False
            """
            self.n = n
            self.chars = chars
            self.model_dict = None  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
            self.vocabulary = None
            self.n_minus_one_dict = None

        def build_model(self, text):
            """populates the instance variable model_dict.

            Args:
            text (str): the text to construct the model from.

            initializes the model dictionary where each element is an ngram,
            initializes the model "n-1 dictionary" where each element is an ngram in size n-1
            initializes the model vocabulary that stores all the words in the language.
            """
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
            """Returns the dictionary class object
            """
            return self.model_dict

        def get_model_window_size(self):
            """Returning the size of the context window (the n in "n-gram")
            """
            return self.n

        def complete_context(self, n, context):
            """
            the method gets a context in size smaller than n as input and completes it to size n
            :param n: the intended size of the context
            :param context: the context to complete
            :return: the completed context
            """
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
            """
            creates a context randomly based on the model's ngrams
            :param n: the ngrams size
            :return: the random context
            """
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
            """
            auxilary function which creates a context based on the given context, where the returned context
            is the first n words in the given context (len(context) > n)
            :param n: the model's n
            :param context: the given context where the returned context should be derived from
            :return: the derived context in size n
            """
            words = context.split()
            prefix = words[0]
            for i in range(1, n):
                prefix = prefix + " " + words[i]
            return prefix

        def generate(self, context=None, n=20):
            """Returns a string of the specified length, generated by applying the language model
            to the specified seed context. If no context is specified the context is sampled
            from the models' contexts distribution. Generation stops before the n'th word if the
            contexts are exhausted. If the length of the specified context exceeds (or equal to)
            the specified n, the method returns the a prefix of length n of the specified context.

            :param context (str): a seed context to start the generated string from. Defaults to None
            :param n (int): the length of the string to be generated.

            :return String. The generated text.

            """
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
            """Returns the smoothed (Laplace) probability of the specified ngram.
            :param: ngram (str): the ngram to have it's probability smoothed
            :return float. The smoothed probability.
            """
            ngram_minus_one = ngram.rsplit(' ', 1)[0]
            counter_minus_1 = 0 if ngram_minus_one not in self.n_minus_one_dict else self.n_minus_one_dict[
                ngram_minus_one]
            counter_ngram = 0 if ngram not in self.model_dict else self.model_dict[ngram]

            return counter_ngram + 1 / (counter_minus_1 + len(self.n_minus_one_dict))

        def evaluate(self, text):
            """Returns the log-likelihood of the specified text to be a product of the model.
            Laplace smoothing applied if necessary.
            :param text (str): Text to evaluate.
            :return Float. The float should reflect the (log) probability.
            """
            text_ngrams = ngrams(re.findall(r'\w+', text.lower()), self.n)
            ngrams_probabilities = {}
            for ngram in text_ngrams:
                ngram = " ".join(ngram)
                ngram_minus_one = ngram.rsplit(' ', 1)[0]
                if ngram not in self.model_dict:
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
    The function removes punctuation chars such as '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    :param text (str): the text to normalize
    :return string. the normalized text.
    """
    punctuations = {}
    for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~''':
        punctuations.add(char)
    punctuations.add('\n')

    normalized_text = ""
    to_lower = text.lower()
    for i in range(len(to_lower)):
        try:
            if to_lower[i] in punctuations:
                try:
                    if to_lower[i + 1] != ' ' and to_lower[i - 1] != ' ':
                        normalized_text += ' '
                except IndexError:
                    continue
            else:
                normalized_text += to_lower[i]
        except UnicodeDecodeError:
            continue
    return normalized_text


def who_am_i():
    """
    :return a dictionary with my name, id number and email. keys=['name', 'id','email']
    """
    return {'name': 'Alona Lasry', 'id': '205567944', 'email': 'alonalas@post.bgu.ac.il'}
