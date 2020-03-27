import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import string
import re

df = pd.read_csv('data.csv')
print(df.columns)
X = df['text'].values
Y=df['polarity'].values
import numpy as np
Y_ = np.array([])
for arr in Y:
    if(arr==0):
        Y_ = np.append(Y_,np.array([1,0]))
    else:
        Y_ = np.append(Y_,np.array([0,1]))
Y = Y_
Y = np.reshape(Y,(int(Y.shape[0]/2),2))
print(Y[0:5])
#removing html tags
clean = re.compile('<.*?>')
df["text"]=df["text"].apply(lambda x: re.sub(clean, '', x))

#lowercase
df["text"]=df["text"].apply(lambda x: x.lower())

#removing punctuation
df["text"]=df["text"].apply(lambda x: re.sub("[^a-zA-Z'! ]+",'', x))

X = df['text'].values

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)

tokenizer_obj = Tokenizer()
total_reviews = X
tokenizer_obj.fit_on_texts(total_reviews) 

# pad sequences
max_length = max([len(s.split()) for s in total_reviews])  
print(total_reviews[2]) 

# define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1

X_train_tokens =  tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)


X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')
print(X_train_pad[0])
print(vocab_size)

EMBEDDING_DIM = 128 

print('Build model...')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model.summary())

model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=1)


print('Testing...')
score, acc = model.evaluate(X_test_pad, y_test, batch_size=128)

print('Test score:', score)
print('Test accuracy:', acc)


test_sample_1 = "fantastic Movie"
test_sample_2 = "Great Movie"
test_sample_3 = "Worst movie"
test_sample_4 = "Bad Movie"
test_sample_5 = "nice movie"
test_sample_6 = "awesome movies"
test_sample_7 = "solid"
test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5,test_sample_6,test_sample_7]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

#predict
print(model.predict(x=test_samples_tokens_pad))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
