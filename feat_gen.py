#!/bin/python
import nltk 
people,stop,facility,company,sports,product,tv,location = set(),set(),set(),set(),set(),set(),set(),set()
def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    for inputfile in ['data/lexicon/firstname.5k']:
        f = open(inputfile, 'r')
        for l in f.readlines():
            people.add(l.strip().lower())
        f.close()
    for inputfile in ['data/lexicon/lastname.5000']:
        f = open(inputfile, 'r')
        for l in f.readlines():
            people.add(l.strip().lower())
        f.close()
    for inputfile in ['data/lexicon/tv.tv_network','data/lexicon/tv.tv_program']:
        f = open(inputfile, 'r')
        for l in f.readlines():
            tv.add(l.strip().lower())
        f.close()
    for inputfile in ['data/lexicon/english.stop']:
        f = open(inputfile, 'r')
        for l in f.readlines():
            stop.add(l.strip().lower())
        f.close()
    for inputfile in ['data/lexicon/sports.sports_league','data/lexicon/sports.sports_team']:
        f = open(inputfile, 'r')
        for l in f.readlines():
            sports.add(l.strip().lower())
        f.close()


    f = open('data/lexicon/book.newspaper', 'r')
    for l in f.readlines():
        company.add(l.strip().lower())
    f.close()

    f = open('data/lexicon/cvg.computer_videogame', 'r')
    for l in f.readlines():
        product.add(l.strip().lower())
    f.close()

    f = open('data/lexicon/venues', 'r')
    for l in f.readlines():
        facility.add(l.strip().lower())
    f.close()

    f = open('data/lexicon/location', 'r')
    c = 1
    for l in f.readlines():
        if c < 5000:
            location.add(l.strip().lower())
            c = c + 1
    f.close()





def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that fire
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word'

    if len(word) >=4:
        ftrs.append("S=" + word[-4:])
    if len(word) >=4:
        ftrs.append("S=" + word[-3:])
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    
    true_shape = ""
    for j in range(len(word)):
        if word[j].isupper():
            true_shape = true_shape + 'X'
        elif word[j].islower():
            true_shape = true_shape + 'x'
        elif word[j].isdigit():
            true_shape = true_shape + 'd'
        else :
            true_shape = true_shape + word[j]

    ftrs.append("TRUE_SHAPE="+true_shape)

    short_word_shape =""
    for k, v in enumerate(word):
        if k == 0 or v != word[k-1]:
            short_word_shape += v
    ftrs.append("SHORT_WORD_SHAPE="+short_word_shape)


    #NEW - ADDED
    
    if word.lower() in stop:
        ftrs.append("STOPWORD")
    else:
        ftrs.append("NOT_STOPWORD")
    if word.lower() in tv:
        ftrs.append("IS_TV")
    else:
        ftrs.append("NOT_TV")
    if word.lower() in company:
        ftrs.append("COMPANY")
    else:
        ftrs.append("NOT_COMPANY")
    if word.lower() in product:
        ftrs.append("PRODUCT")
    else:
        ftrs.append("NOT_PRODUCT")
    if word.lower() in location:
        ftrs.append("LOCATION")
    else:
        ftrs.append("NOT_LOCATION")
    if word.lower() in facility:
        ftrs.append("FACILITY")
    else:
        ftrs.append("NOT_FACILITY")
    if word.lower() in people:
        ftrs.append("people")
    else:
        ftrs.append("NOT_people")

    if word.lower() in sports:
        ftrs.append("SPORTS")
    else:
        ftrs.append("NOT_SPORTS")




    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "_main_":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)