"""
Importing essential libraries
"""
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, BatchNormalization
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import Adam
import string
import re
tensorflow.compat.v1.disable_eager_execution()

"""
Reading data from the data.csv file which is generated using 'generate_data.py'.
"""
df = pd.read_csv('data.csv')
X = df['text'].values
Y=df['polarity'].values 

"""
Building a training and validation set
"""
X = df['text'].values
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.1,stratify=Y)

"""
Creating a Tokenizer object and using 'fit_on_texts' function to generate token values for top 50,000 words that occur in the corpus from the corpus.
"""
tokenizer_obj = Tokenizer(num_words=80000)
tokenizer_obj.fit_on_texts(X_train)
np.save("X_train.npy",X_train) 

"""
Calculating the maximum length (most number of words) of a tweet among the corpus. 
"""
total_reviews = X
max_length = max([len(s.split()) for s in total_reviews])  

"""
Calculating the vocab size which will be used in embedding layer.
"""
vocab_size = len(tokenizer_obj.word_index) + 1

"""
Converting text to vectors of size 'max_length' having token values.
"""
X_train_tokens =  tokenizer_obj.texts_to_sequences(X_train)
X_val_tokens = tokenizer_obj.texts_to_sequences(X_val)
X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_val_pad = pad_sequences(X_val_tokens, maxlen=max_length, padding='post')


"""
Building model
"""
EMBEDDING_DIM = 64

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(LSTM(units=64, dropout=0.3, return_sequences=True, recurrent_dropout=0.3))
model.add(BatchNormalization())
model.add(LSTM(units=32,  dropout=0.3, return_sequences=True, recurrent_dropout=0.3))
model.add(BatchNormalization())
model.add(LSTM(units=32,  dropout=0.3, recurrent_dropout=0.3))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.0003)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

print('Summary of the built model : ')
print(model.summary())

"""
Training the model
"""
model.fit(X_train_pad, y_train, batch_size=256, epochs=10, validation_data=(X_val_pad, y_val), verbose=1)

"""
Validating the model
"""
score, acc = model.evaluate(X_val_pad, y_val)

print('Validation score:', score)
print('Validation accuracy:', acc)

"""
Saving model architecture and weights
"""
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

