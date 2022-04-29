
import pandas as pd

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
   
from nltk.stem import WordNetLemmatizer

# -- PARAMETERS --
CUSTOM_SW = ["semst"] # TODO: add to actual stopwords
VOCAB_PATH = "combined_claim_vocab.csv"
USE_LEMMATIZER = True

test_string = "ISIS ISIS says you claims foley"

# takes [string], returns lowercased & lemmatized [string]
def lem_str(in_string):
    out_str = [""]
    lemmatizer = WordNetLemmatizer()
    for word in in_string.split():
        out_str[0] += (lemmatizer.lemmatize(word.lower()) + " ")
    return out_str

# takes vocab & returns bow_vectorizer
def load_vectorizer():
    # define stopwords
    sw = text.ENGLISH_STOP_WORDS.union(["book"])
    # read vocabulary
    vocab_df = pd.read_csv(VOCAB_PATH, header=None)
    vocab_df = vocab_df.drop([0])

    bow_vectorizer = CountVectorizer(
        stop_words=sw,
        max_features=5000,
        vocabulary=vocab_df[1]
    )
    print(bow_vectorizer.vocabulary[:-20])

    return bow_vectorizer

# takes string and yeets it through the BoW pipeline
def create_bow(in_string):
    bow_vectorizer = load_vectorizer()
    if USE_LEMMATIZER == True:
        bow = bow_vectorizer.fit_transform(lem_str(in_string), y=None)
    else:
        bow = bow_vectorizer.fit_transform([in_string], y=None)
    return bow

# TODO: 
# test on new data frame

def create_idf(bow_vec):
    tfreq_vec = TfidfTransformer(use_idf=False).fit(bow_vec)
    tfreq = tfreq_vec.transform(bow_vec)
    return tfreq

print(create_idf(create_bow(test_string)).toarray()[0])

# TODO für Raphael/Ruben:
# - Warum das vocab script nicht genau 5k groß ist
#   - Falsche Lemmatization?
# - Warum wörter wie "u" im vocab vorkommen