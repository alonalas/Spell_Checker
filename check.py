import re
from collections import Counter
from nltk import ngrams

#my_dict = {}
text = open('big.txt').read()
my_ngrams = ngrams(re.findall(r'\w+', text.lower()), 3)
#word_list =re.findall(r'\w+', text.lower())
#WORDS = Counter(word_list)
my_dict = dict(Counter(my_ngrams))

print(my_ngrams)