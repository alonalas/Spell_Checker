import re
from collections import Counter
from nltk import ngrams
import random
import math
from ex1 import Spell_Checker as sc
import ex1
import spelling_confusion_matrices


spell_checker = sc()
spell_checker.lm = sc.Language_Model(3,False)
spell_checker.add_error_tables(spelling_confusion_matrices.error_tables)
text = open("big.txt", "rb").read().decode("utf-8")
spell_checker.lm.build_model(ex1.normalize_text(text))
print("hey")

print(spell_checker.spell_check("two of them apple",0.95))

