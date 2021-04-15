import re
from collections import Counter
from nltk import ngrams
import random
import math
from ex1 import Spell_Checker as sc
import spelling_confusion_matrices

spelling_confusion_matrices
spell_checker = sc()
spell_checker.lm = sc.Language_Model(4,False)
spell_checker.add_error_tables(spelling_confusion_matrices.error_tables)
text = open("corpus.data", "rb").read().decode("utf-8")
spell_checker.lm.build_model(text)
#spell_checker.lm.build_model(open('trump_historical_tweets.txt', encoding="utf8").read())
# spell_checker.lm.build_model(open('big.txt').read())
# #context_built = spell_checker.lm.generate(context="i like americans",n=15)
# print(math.pow(10,spell_checker.evaluate("watching TV is important")))

print(spell_checker.spell_check("todat i went to school",0.95))


