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

    lemmatized_stop_words = []
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
    sp.save_npz(OUTPUT_DIR / output_file, dtm)

    return vect


def cleanTest(vect, test_docs, output_file):
    dtm = vect.transform(test_docs)
    logger.info("\t\tTest DTM shape: %s\n" % (dtm.shape,))
    sp.save_npz(OUTPUT_DIR / output_file, dtm)


def printCnt(docs, labels):
    logger.info("\t\tNumber of training reviews: %s" % len(docs))
    logger.info("\t\tPositive: %s; Negative: %s\n" % (sum(labels),
                                                      len(labels) - sum(
                                                          labels)))


if __name__ == '__main__':
    train_path = INPUT_DIR / 'train'
    test_path = INPUT_DIR / 'test'

    logger.info("\tReading raw training data:")
    train_reviews, train_labels = getData(train_path)
    printCnt(train_reviews, train_labels)

    logger.info("\tReading raw test data:")
    test_reviews, test_labels = getData(test_path)
    printCnt(test_reviews, test_labels)

    logger.info("\tFitting and transforming training text to training DTM")
    createDirs(OUTPUT_DIR)
    train_vect = cleanTrain(train_reviews, "X_train", min_df=10)

    # Save feature names:
    feature_names = train_vect.get_feature_names()
    with open(OUTPUT_DIR / 'feature_names.txt', 'w') as f:
        f.write("\n".join(feature_names))
    logger.info("\t\tFeature names saved.\n")

    logger.info("\tTransforming test text to test DTM")
    cleanTest(train_vect, test_reviews, "X_test")

    save_target = lambda path, docs: np.save(path, np.array(docs).reshape(-1,
                                                                          1),
                                             allow_pickle=False)
    save_target(OUTPUT_DIR / "y_train", train_labels)
    save_target(OUTPUT_DIR / "y_test", test_labels)

    logger.info("\tFinished raw text cleaning, DTMs and targets saved to %s" %
                OUTPUT_DIR.absolute())
    logger.info("Raw text cleaning finished.")
