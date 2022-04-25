from os import rename
import pandas as pd
import numpy as np

# import both datasets
headline_df = pd.read_csv('train_stances.csv', header=None)
tweet_df = pd.read_csv('testdata-taskA-all-annotations.txt', sep='\t', header=None)

# Change the frames to be 1:1
tweet_df.drop([0,1,3,4,5], inplace=True, axis=1)
tweet_df = tweet_df.rename(columns={2:0})
tweet_df = tweet_df.drop([0,0])
headline_df = headline_df.drop([0,0])
headline_df.drop([1,2], inplace=True, axis=1)
# index to the texts (TODO: select the right coloum)
tweets = tweet_df
headlines= headline_df

# merge texts
frames = [tweets,headlines]
df = pd.concat(frames)
print(df)


"""
wenn das nicht geht, probier sowas wie 
combined = tweets[index].join(headlines[index2])
"""
# write as csv
df.to_csv('data_set.csv')