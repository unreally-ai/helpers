import pandas as pd
from nltk.stem import WordNetLemmatizer

#my_data = np.genfromtxt('../vocab_script/vocab.csv', delimiter=',')
#my_data = np.loadtxt('../vocab_script/vocab.csv', delimiter=',')

my_data = pd.read_csv('../vocab_script/vocab.csv', header=None)
my_data = my_data.drop([0,0])
lemmatizer = WordNetLemmatizer()
print(my_data.head())
def createList(n):
    lst = []
    for i in range(n):
        lst.append(0)
    return(lst)

lst = createList(5000)

def bow(data,claim):
    for word in claim.split():
        word = lemmatizer.lemmatize(word.lower())
        print(word)
        i = -1
        for row in data[1]:
            i += 1
            if row == word:
                lst[i] += 1
    return lst

tf = bow(my_data,'isi isi 1')
print(tf[:20])
print(len(tf))