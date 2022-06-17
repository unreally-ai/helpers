""" 
                            _ _        
 _   _ _ __  _ __ ___  __ _| | |_   _  
| | | | '_ \| '__/ _ \/ _` | | | | | | 
| |_| | | | | | |  __/ (_| | | | |_| | 
 \__,_|_| |_|_|  \___|\__,_|_|_|\__, | 
                                |___/  live pipeline!
"""
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer

# ----------------------------- PARAMETERS -----------------------------
CUSTOM_SW = ["semst","u"] # TODO: add to actual stopwords
VOCAB_PATH = "vocab_headlines.csv"
BODY_PATH = "vocab_bodies.csv"
USE_LEMMATIZER = True

# TODO: select headlines & body out of dataset 
test_string = "ISIS ISIS us us says you claims foley"
test_body = "apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley "
vocab_df = pd.read_csv(VOCAB_PATH, header=None)

# ----------------------------- BOW VECTORIZER PIPELINE -----------------------------
# takes [string], returns lowercased & lemmatized [string]
def lem_str(in_string):
    out_str = [""]
    lemmatizer = WordNetLemmatizer()
    for word in in_string.split():
        out_str[0] += (lemmatizer.lemmatize(word.lower()) + " ")
    
    return out_str

# takes vocab & returns bow_vectorizer
def load_vectorizer(path):
    # define stopwords
    sw = text.ENGLISH_STOP_WORDS.union(["book"])
    # read vocabulary
    vocab_df = pd.read_csv(path, header=None)
    vocab_df = vocab_df.drop([0])

    bow_vectorizer = CountVectorizer(
        stop_words=sw,
        max_features=5000,
        vocabulary=vocab_df[1]
    )


    return bow_vectorizer

# takes string & path to vocab and yeets it through the BoW pipeline
def create_bow(in_string, path):
    bow_vectorizer = load_vectorizer(path)
    if USE_LEMMATIZER == True:
        bow = bow_vectorizer.fit_transform(lem_str(in_string), y=None)
    else:
        bow = bow_vectorizer.fit_transform([in_string], y=None)

    return bow


# ----------------------------- TF-IDF PIPELINE -----------------------------

def create_tf(bow_vec):
    tfreq_vec = TfidfTransformer(use_idf=False).fit(bow_vec)
    tfreq = tfreq_vec.transform(bow_vec)

    return tfreq

# TODO figure out

def create_tfidf(bow):
    tfreq_vec = TfidfTransformer(use_idf=True)
    tfreq = tfreq_vec.fit_transform(bow)

    return tfreq

# -------------- GENEREATE 10001 VECTOR --------------


def yeet2vec(head, body):

    # get our sub-vectores
    claim_tf = create_tf(create_bow(head, VOCAB_PATH))
    body_tf = create_tf(create_bow(body, BODY_PATH))
    
    claim_tfidf = create_tfidf(create_bow(head, VOCAB_PATH))
    bodie_tfidf = create_tfidf(create_bow(body, BODY_PATH))

    print("  - created sub-vectors ✅")

    # tasty cosine similarity
    c_sim = cosine_similarity(claim_tfidf ,bodie_tfidf)

    # do the yeeting
    # HERE IS SOME ERRROR WITH CONCAT
    tenk = pd.concat([claim_tf, c_sim, body_tf],axis=1)
    tenk = tenk.to_numpy()
    tenk = torch.from_numpy(tenk)

    print("  - created 10k vector ✅")
    return tenk

# yeets strings through pipeline, outputs finished 10k vector
vector = yeet2vec(test_string, test_body)

# -------------- LOAD ML MODEL --------------

# layer size
in_dim = 10001
hidden_dim = 100
out_dim = 4

# define model class
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(NN, self).__init__()
        # define layers
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, out_dim)
    
    # applies layers with sample x
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# load the chungifier5000
print("🤖 model go brr ...")
model = NN(in_dim, hidden_dim, out_dim)
model.load_state_dict(torch.load('chungifier5000.pth'))

# -------------- PREDICT ON DATA --------------

output = model(vector.float()) # does this work?!

# TODO: figure out how to convert model output to stance
output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

print(output)