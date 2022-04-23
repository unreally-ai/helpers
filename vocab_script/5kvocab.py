from nltk.stem import WordNetLemmatizer 
import sys
import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
import glob
import os

# --- USER PARAMETERS ---

custom_stopwords = ["semst", "im"]
# column of the text
index = 0
# csv or txt
data_type = "csv" 
##lemmetizer
lemmatizer = WordNetLemmatizer()

# --- DEFINE PIPELINE FUNCTIONS ---

# takes in string & returns a cleaned string of all non-stop-words
def preprocess(text):
    sw = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text).lower()
    s = ""
    for word in text.split():
        if word not in sw and word not in custom_stopwords:
                s += (lemmatizer.lemmatize(word) + " ")
    return s

# Takes array of dataframes, returns df with top5k dictionary
def multidf_vocab(df_arr):
    # create array of cleaned strings
    vocab = []
    for df in df_arr:
        for i in range(len(df)-1):
            vocab.append(preprocess(df[index][i]))
    vocab_df = pd.DataFrame(vocab)
    # how do I use counter without turning the vocab array into a df first?
    # count appearance of each word & create frequency dataframe
    return vocab_df


# takes in cleaned text df, return top 5k frequency df
def tf5k(processed_df):
    counter = Counter(" ".join(processed_df[0]).split()).most_common(5000)
    counter_df = pd.DataFrame(counter)
    return counter_df

# try reading each df
def read_dfs(directory):
    dfs = []
    for filename in directory:
        if data_type == "csv":
            df = pd.read_csv(filename, sep="\t", header=None)
        elif data_type == "txt":
            df = pd.read_csv(filename, sep="\t", header=None)
        dfs.append(df)
        print("dataframes read...")
        return dfs
  

# call all of the pipeline functions
def pipeline(dfs):
    # --PIPELINE---
    data = multidf_vocab(dfs)
    top5k = tf5k(data)
        
    # save to file
    top5k.to_csv('./vocab.csv')
    print("created top 5k dictionary")


# --- START ---

# catch wrong usage
if len(sys.argv) < 2 or len(sys.argv) > 2:
    print("usage:\n python3 5kvocab.py <folder>")
else:
    path = sys.argv[1]
    select = f"*.{data_type}"
    directory = glob.glob(os.path.join(path , select)) 
    print("dir:", directory)
    
    dfs = read_dfs(directory)
    pipeline(dfs)
