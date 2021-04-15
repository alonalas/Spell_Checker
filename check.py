import re
from collections import Counter
from nltk import ngrams
import random
from ex1 import Spell_Checker as sc

lm = sc.Language_Model(5,False)
lm.build_model(open('trump_historical_tweets.txt', encoding="utf8").read())
#lm.build_model(open('big.txt').read())
print(lm.generate(context="americans are the most",n=15))




