"""
Importing necessary libraries.
"""
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import re
import string
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""
Creating dataframe using pandas from the data file 'training.1600000.processed.noemoticon.csv' obtained from url 'https://www.kaggle.com/kazanova/sentiment140'.
"""
df = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1")

"""
Following piece of code creates 'X' containing 6,00,000 tweets (3,00,000 positive and 3,00,000 negative) and 'Y' which contains labels (0 - negative, 1 - positive)
"""
X = df["@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"].values
X1 = X[:300000]
X2 = X[1250000:1550000]
X_ = np.array((600001))
X_ = np.append(X_,X1)
X_ = np.append(X_,X2)
X = X_
X = np.delete(X,[0])
Y = df['0'].values
Y1 = Y[:300000]
Y2 = Y[1250000:1550000]
Y_ = np.array((600001))
Y_ = np.append(Y_,Y1)
Y_ = np.append(Y_,Y2)
Y = Y_
Y = np.delete(Y,[0])
tok_obj = Tokenizer()
remover = []
X_mod = []
count = 0
for text in X:
    if type(text) is float:
        remover.append(count)
    elif(not(text.lower)):
        remover.append(count)
    else:
        X_mod.append(text)
    count = count + 1
Y = np.delete(Y,remover)
for val in range(0,Y.shape[0]):
    if Y[val] == 4:
        Y[val] = 1
X = np.array(X_mod)

"""
Following piece of code creates a dataframe which has a column for tweets text and another for labels.
"""
data = pd.DataFrame()
data['text'] = X.tolist()
data['polarity'] = Y.tolist()

"""
Following piece of code preprocesses each tweet's text. The steps for preprocessing are : 
-> Remove links.
-> Convert text to lowercase.
-> Remove punctuations from the text.
"""
clean = re.compile('<.*?>')
data["text"]=data["text"].apply(lambda x: re.sub(clean, '', x))
data["text"]=data["text"].apply(lambda x: x.lower())
data["text"]=data["text"].apply(lambda x: re.sub("[^a-zA-Z'! ]+",'', x))

"""
Saving the dataframe as csv.
"""
data.to_csv('data.csv')