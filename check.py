import re
from collections import Counter
from nltk import ngrams
import random

my_dict = {}
text = open('big.txt').read()
my_ngrams = ngrams(re.findall(r'\w+', text.lower()), 3)
word_list =re.findall(r'\w+', text.lower())
WORDS = Counter(word_list)
my_dict = dict(Counter(my_ngrams))
for old_key in list(my_dict.keys()):
    my_dict[" ".join(old_key)] = my_dict[old_key]
    del my_dict[old_key]

#
# print(my_ngrams)
# context = "hello my name is alona and today is the memorial day"
# words = context.split()
# prefix = words[0]
# for i in range(1,1):
#     prefix = prefix + " " + words[i]
# print(prefix)
given_n = 20
self_n = 3
n_minus_one_dict = dict(Counter(ngrams(re.findall(r'\w+', text.lower()), 2)))
for old_key in list(n_minus_one_dict.keys()):
    n_minus_one_dict[" ".join(old_key)] = n_minus_one_dict[old_key]
    del n_minus_one_dict[old_key]


random_ngram = random.choices(list(my_dict.keys()), weights=list(my_dict.values()), k=1)[0]
context = random_ngram
if self_n < given_n:
    for i in range (0,given_n-self_n):
        end = random_ngram.split(' ', 1)[1]
        sub_dict = {k: v for k, v in my_dict.items() if k.startswith(end + " ")}
        addition = random.choices(list(sub_dict.keys()), weights=list(sub_dict.values()),k=1)[0].split()[-1]
        context = context + " " + addition
        random_ngram = end + " " + addition
    print(context)
else:
    print("blala")