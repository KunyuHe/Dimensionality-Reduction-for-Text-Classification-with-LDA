from pathlib import Path
import re
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm

PATH = Path(r'..\data\aclImdb')

TRAIN_PATH = PATH / 'train'
TEST_PATH = PATH / 'test'

STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['look', 'seem', 'say', 'thing', 'know', 'may', 'feel',
                   'want', 'ever', 'actually', 'take', 'come', 'become', 'also',
                   'much', 'use'])

NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def getDocuments(path):
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


def preprocess(input_path, output_path):
    documents, labels = getDocuments(input_path)
    output = []
    allowed_postags = ('NOUN', 'ADJ', 'VERB', 'ADV')

    for i in tqdm(range(len(documents))):
        # Strip tags
        s = re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", documents[i])

        # Convert document into a list of lowercase tokens, ignoring tokens that
        # are too short or too long
        s = simple_preprocess(s, deacc=True, min_len=2, max_len=15)

        # Remove stopwords
        s = [w for w in s if w not in STOP_WORDS]

        # Lemmatization
        doc = NLP(" ".join(s))
        s = [token.lemma_ for token in doc if token.pos_ in allowed_postags]

        output.append(s)

    return output






