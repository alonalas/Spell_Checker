import re
from collections import Counter
from nltk import ngrams
import random
import math
from ex1 import Spell_Checker as sc
import spelling_confusion_matrices


spell_checker = sc()
spell_checker.lm = sc.Language_Model(4,False)
spell_checker.add_error_tables(spelling_confusion_matrices.error_tables)
text = open("corpus.data", "rb").read().decode("utf-8")
spell_checker.lm.build_model(text)
#
print(spell_checker.spell_check("i am making two errourss in purpose",0.95))
#



