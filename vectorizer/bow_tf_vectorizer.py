from bz2 import compress
import sys
import pandas as pd
import numpy as np

import torch

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import WordNetLemmatizer

# -- PARAMETERS --
CUSTOM_SW = ["semst","u"] # TODO: add to actual stopwords
VOCAB_PATH = "vocab_script/vocab_headlines.csv"
BODY_PATH = "vocab_script/vocab_bodies.csv"
USE_LEMMATIZER = True

test_string = "ISIS ISIS us us says you claims foley"
test_body = "apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley apple said you would report one woman and Isis us says your claim was foley "
vocab_df = pd.read_csv(VOCAB_PATH, header=None)
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

# takes string and yeets it through the BoW pipeline
def create_bow(in_string, path):
    bow_vectorizer = load_vectorizer(path)
    if USE_LEMMATIZER == True:
        bow = bow_vectorizer.fit_transform(lem_str(in_string), y=None)
    else:
        bow = bow_vectorizer.fit_transform([in_string], y=None)
    return bow

# TODO: 
# test on new data frame

def create_tf(bow_vec):
    tfreq_vec = TfidfTransformer(use_idf=False).fit(bow_vec)
    tfreq = tfreq_vec.transform(bow_vec)
    return tfreq

# TODO figure out

def create_tfidf(bow):
    tfreq_vec = TfidfTransformer(use_idf=True)
    tfreq = tfreq_vec.fit_transform(bow)
    return tfreq

labels = {
        "agree": torch.tensor([0]),
        "disagree": torch.tensor([1]),
        "discuss": torch.tensor([2]),
        "unrelated": torch.tensor([3]),
    }

def find_label(item):
    print(labels[item])
    return labels[item]

data = pd.read_csv("/Users/vince/unreally/helpers-main/train_data.csv")
terror = []
for i in range(len(data)):
    claim_tf = create_tf(create_bow(data['head'][i],VOCAB_PATH))
    body_tf = create_tf(create_bow(data['body'][i],BODY_PATH)) #'head'
    claims = create_tfidf(create_bow(data['head'][i],VOCAB_PATH))
    bodies = create_tfidf(create_bow(data['body'][i],BODY_PATH))
    c_similarity = cosine_similarity(claims,bodies)

    claim_df = pd.DataFrame(claim_tf.toarray()) 
    body_df = pd.DataFrame(body_tf.toarray())
    c_similarity_df = pd.DataFrame(c_similarity)

    almighty = pd.concat([claim_df,c_similarity_df,body_df],axis=1)
    almighty = almighty.to_numpy()
    almighty = torch.from_numpy(almighty)
    
    terror.append([almighty,find_label(data['stance'][i]),data['ID'][i]])

df = pd.DataFrame (terror, columns = ['vector','stance','ID'])
# create csv file
print(df['vector'][2])
print(df['vector'][2][0][30])
np.set_printoptions(threshold=sys.maxsize)
df.to_pickle('terror4.3.pkl')
print(df)