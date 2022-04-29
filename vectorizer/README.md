# vectorizer
this directory contains scripts used for text vectorization

## bow_idf_vectorizer
> Requires a dictionary. The path can be changed at the top of the code under _parameters_.

Currently, it simply runs and computes the IDF (not tfidf!!) value of a string defined
in the code. By setting USE_LEMMATIZER = False, the program can run without lemmatization.
Note that this of course requires a non-lemmatized dictionary.
