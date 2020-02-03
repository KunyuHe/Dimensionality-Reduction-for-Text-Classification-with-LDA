import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import spacy
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from utils import createDirs, createLogger

INPUT_DIR = Path(r'../data/aclImdb/')
OUTPUT_DIR = Path(r'../data/clean/')
LOG_DIR = Path(r'../logs/clean/')

logger = createLogger(LOG_DIR, "clean")
logger.info("Logger created, logging to %s" % LOG_DIR.absolute())


def getData(path):
    """
    Get IMDb movie reviews and corresponding labels.

    Inputs:
        - path (str): IMDb movie review directory

    Outputs:
        ([str], [int]): list of movie reviews, list of sentiment labels
    """
    texts, labels = [], []

    for review in ['pos', 'neg']:
        for p in (path / review).glob('*.txt'):
            with open(p, encoding='utf-8') as f:
                texts.append(f.read())
            if review == 'pos':
                labels.append(1)
            else:
                labels.append(0)

    return texts, labels


def cleanTrain(train_docs, output_file, min_df=10):
    """
    Transform training texts into document-term-matrix (DTM), where each row is
    a document, each column is a term, and each element represents frequency of
    the term in that document. Lemmatize, remove stop words, and apply bi-gram
    transformation. Save the training DTM.

    Inputs:
        - train_docs ([str]): training documents, each element is a document
            represented as string
        - output_file (str): path to save the training DTM as a .npz
        - min_df (int): minimum frequency for a term to be included in the DTM

    Outputs:
        (sklearn.feature_extraction.text.CountVectorizer) transformer to be
            applied on the test texts
    """
    docs = [doc.replace("<br />", " ") for doc in train_docs]

    # Lemmatization
    # python -m spacy download en_core_web_sm
    en_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    postags = ["NOUN", "ADJ", "ADV", "VERB"]

    regexp = re.compile('(?u)\\b\\w\\w+\\b')
    old_tokenizer = en_nlp.tokenizer
    en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
        regexp.findall(string))
    custom_tokenizer = lambda doc: [token.lemma_ for token in en_nlp(doc)
                                    if token.pos_ in postags]

    # Remove stop words
    with open(Path(r'../data/') / "extended_stopwords.txt", "r") as f:
        stop_words_extend = f.readline().split(",")
    stop_words = ENGLISH_STOP_WORDS.union(set(stop_words_extend))

    lemmatized_stop_words = stop_words_extend[:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for stop_word in stop_words:
            lemmatized_stop_words.extend(custom_tokenizer(stop_word))
    lemmatized_stop_words = set(lemmatized_stop_words)

    # Bigram with `min_df`
    vect = CountVectorizer(tokenizer=custom_tokenizer,
                           stop_words=lemmatized_stop_words,
                           ngram_range=(1, 2),
                           min_df=min_df)
    dtm = vect.fit_transform(docs)
    logger.info("\t\tTraining DTM shape: %s" % (dtm.shape,))
    sp.save_npz(output_file, dtm)

    return vect


def cleanTest(vect, test_docs, output_file):
    """
    Transform test texts into DTM with the training CountVectorizer. Save the
    test DTM.

    Inputs:
        - vect (sklearn.feature_extraction.text.CountVectorizer):
            CountVectorizer fitted on the training set
        - test_docs ([str]): test documents
        - output_file (str): path to save the test DTM as a .npz
    """
    dtm = vect.transform(test_docs)
    logger.info("\t\tTest DTM shape: %s\n" % (dtm.shape,))
    sp.save_npz(output_file, dtm)


def printCnt(docs, labels, train=True):
    """
    Log the size and balance of training and test sets.

    Inputs:
        - docs ([str]): training or test documents
        - labels ([int]): sentiment class of the documents
        - train (bool): True we are working with the training documents; False
            otherwise

    Output:
        (None)
    """
    logger.info("\t\t%s data summary:" % ["Test", "Training"][int(train)])
    logger.info("\t\t\tNumber of training reviews - %s" % len(docs))
    logger.info("\t\t\tPositive - %s; Negative - %s\n" % (sum(labels),
                                                         len(labels) - sum(
                                                             labels)))


def go(seed, up_to, min_df):
    train_path = INPUT_DIR / 'train'
    test_path = INPUT_DIR / 'test'

    logger.info("\tReading raw data:")
    train_reviews, train_labels = getData(train_path)
    test_reviews, test_labels = getData(test_path)

    # Get a subset of training data, move the rest to test
    np.random.seed(seed)
    train_inds = np.zeros((len(train_reviews),), dtype=bool)
    train = np.random.choice(range(len(train_reviews)), up_to, replace=False)
    train_inds[train] = True

    test_reviews += np.array(train_reviews)[~train_inds].tolist()
    test_labels += np.array(train_labels)[~train_inds].tolist()
    train_reviews = np.array(train_reviews)[train_inds].tolist()
    train_labels = np.array(train_labels)[train_inds].tolist()

    printCnt(train_reviews, train_labels)
    printCnt(test_reviews, test_labels, train=False)


    output_dir = OUTPUT_DIR / ('subset_%s/' % up_to)

    logger.info("\tFitting and transforming training text to training DTM")
    createDirs(output_dir)
    train_vect = cleanTrain(train_reviews, output_dir / "X_train", min_df=min_df)

    # Save feature names:
    feature_names = train_vect.get_feature_names()
    with open(output_dir / 'feature_names.txt', 'w') as f:
        f.write("\n".join(feature_names))
    logger.info("\t\tFeature names saved.\n")

    logger.info("\tTransforming test text to test DTM")
    cleanTest(train_vect, test_reviews, output_dir / "X_test")

    save_target = lambda path, docs: np.save(path, np.array(docs),
                                             allow_pickle=False)
    save_target(output_dir / "y_train", train_labels)
    save_target(output_dir / "y_test", test_labels)

    logger.info("\tFinished raw text cleaning, DTMs and targets saved to %s" %
                output_dir.absolute())
    logger.info("Raw text cleaning finished.")


if __name__ == '__main__':
    desc = ("Transform raw text data into document-term matrix.")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--up_to', dest='up_to', type=int, default=25000,
                        help="Number of training samples allowed.")
    parser.add_argument('--min_df', dest='min_df', type=int, default=10,
                        help="Minimum frequency of a term to be included.")
    parser.add_argument('--seed', dest='seed', type=int, default=123,
                        help="Seed for NumPy random module.")
    parser.add_argument('--all', dest='all', type=bool, default=False,
                        help="Whether to run for all deciles.")
    args = parser.parse_args()

    if args.all:
        for up_to in np.linspace(1250, 25000, 20):
            go(args.seed, int(up_to), args.min_df)
    else:
        go(args.seed, args.up_to, args.min_df)
