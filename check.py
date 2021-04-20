from datetime import datetime

from ex1 import Spell_Checker as sc
import ex1
import spelling_confusion_matrices
import time
#
# spell_checker = sc()
# spell_checker.lm = sc.Language_Model(3,False)
# spell_checker.add_error_tables(spelling_confusion_matrices.error_tables)
# text = open("big.txt", "rb").read().decode("utf-8")
# spell_checker.lm.build_model(ex1.normalize_text(text))

#---------------------------- Tests ----------------------------#
alpha = 0.95
sentences_wrong_words = [
    ("we also enjoyed the popular acress", "we also enjoyed the popular actress"),
    ("swim acress the pool", "swim across the pool"),

    ("chronic inflammation are caused by infection with a specific organism all having the common karacter",
     "chronic inflammation are caused by infection with a specific organism all having the common character"),

    ("two of thew apples", "two of the apples"),

    ("the vegetarian restaurant serves good fod", "the vegetarian restaurant serves good food"),
    ("he vegetarian restaurant serves good food", "the vegetarian restaurant serves good food"),
    ("the vegetarian restaurant serves god food", "the vegetarian restaurant serves good food"),

    ("i could not eccept such conditions", "i could not accept such conditions"),
    ("i like everything accept that", "i like everything except that"),
    ("talk to no one eccept me", "talk to no one except me"),

    ("the disappearance of the buffalo the main fod supply of the wild indians", "the disappearance of the buffalo the main food supply of the wild indians"),

    ("abondon","abandon"),
    ("abotu","about"),
    ("leutenant","lieutenant"),
    ("recident","resident"),
    ("same mischievious","same mischievous"),
    ("wroet","wrote")
 ]

#sentences_wrong_words = [("the famous acress ate the apple", "the famous actress ate the apple")]



def correcting_senteces(sc):
    num_of_good_corrections = 0
    for wrong_correct_tpl in sentences_wrong_words:
        wrong, correct = wrong_correct_tpl
        start = datetime.now()
        print(f'Wrong sentence: {wrong}')
        try:
            correction = sc.spell_check(wrong, alpha)
            print(f'Correction: {correction}')
            if correction == correct:
                print('Good')
                num_of_good_corrections += 1
            else:
                print('Bad')
            print(f'Correction time: {datetime.now() - start}\n')
        except TypeError: print('Exception thrown')
    print(f'\nNum of good corrections is {num_of_good_corrections} out of {len(sentences_wrong_words)}')


chars = False
n = 3
print(f'\nN-gram is {n}\nChars is {chars}\n')
#--- big.txt tests ---#

sc = ex1.Spell_Checker()
lm = sc.Language_Model(n = n, chars=chars)
sc.add_language_model(lm)
from spelling_confusion_matrices import error_tables
sc.add_error_tables(error_tables)
print('\n#--------------------- big.txt ---------------------#')
big = open('big.txt', 'r').read()
start = datetime.now()
lm.build_model(big)
end = datetime.now()
print(f'Building the model took:   {end - start}\n')
correcting_senteces(sc)

#--- corpus.data tests ---#
sc = ex1.Spell_Checker()
lm = sc.Language_Model(n = n, chars = chars)
sc.add_language_model(lm)
from spelling_confusion_matrices import error_tables
sc.add_error_tables(error_tables)

print('\n#--------------------- corpus.data ---------------------#')
corpus = open('corpus.data', 'r').read()
# corpus = ' '.join(corpus.split('<s>'))
lm.build_model(corpus)
end = datetime.now()
print(f'Building the model took:   {end - start}\n')
correcting_senteces(sc)