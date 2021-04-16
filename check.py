import re
from collections import Counter
from nltk import ngrams
import random
import math
from ex1 import Spell_Checker as sc
import spelling_confusion_matrices

# spelling_confusion_matrices
# spell_checker = sc()
# spell_checker.lm = sc.Language_Model(4,False)
# spell_checker.add_error_tables(spelling_confusion_matrices.error_tables)
# text = open("corpus.data", "rb").read().decode("utf-8")
# spell_checker.lm.build_model(text)
#
# print(spell_checker.spell_check("todat i went to school",0.95))

def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    inserts = [(L + c + R, "insertion " + L[0:1] + c) for L, R in splits for c in letters]
    print(inserts)
    # for L,R in splits:
    #     for c in letters:
    #         if R:
    #             print("replace " , L + c + R[1:], R[0:1] + c)
    #             #inserts.append(L + c + R)
    #
    # # the_set = set(deletes + transposes + replaces + inserts)
    # # return the_set
    # return replaces

print(edits1("todat"))


