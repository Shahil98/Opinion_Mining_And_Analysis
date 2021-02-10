"""
Importing necessary libraires.
""" 
import tweepy
import json
import re
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import model_from_json
import random
from flask import Flask,render_template,url_for,request
import numpy as np
import emoji

app = Flask(__name__)

"""
Function to render page http://127.0.0.1:5000/
"""
@app.route('/')
def hello(st=''):
    print("HOME")
    return render_template('home.html',title='home')

"""
Function to render page http://127.0.0.1:5000/analysis
"""
@app.route('/analysis',methods=['POST','GET','OPTIONS'])   
def analysis():
    
    """
    Taking search query into the variable 'key'.
    """
    key=request.form['InputText']

    """
    Performing authentication to access twitter's data.
    (Use twitter developer credentials below and uncomment the following piece commented code).
    """
    """
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    auth.set_access_token(access_token, access_token_secret)
    """

    """
    Creating an api object using tweepy.
    """
    api = tweepy.API (auth) 

    """
    Fetching tweets and storing them in results array. 'num' variable denotes the number of tweets to be fetched.
    """
    results = [] 
    num = 50 
    for tweet in tweepy.Cursor (api.search, q = key, lang = "en").items(num): 
        results.append(tweet)
    
    """
    Creating a pandas dataframe to capture tweet information.
    """
    dataset=pd.DataFrame()
    dataset["tweet_id"]=pd.Series([tweet.id for tweet in results])
    dataset["username"]=pd.Series([tweet.author.screen_name for tweet in results])
    dataset["text"]=pd.Series([tweet.text for tweet in results])
    dataset["followers"]=pd.Series([tweet.author.followers_count for tweet in results])
    dataset["hashtags"]=pd.Series([tweet.entities.get('hashtags') for tweet in results])
    dataset["emojis"]=pd.Series([','.join(c for c in tweet.text if c in emoji.UNICODE_EMOJI) for tweet in results])

    """
    Following piece of code is used to generate wordcloud of the hashtags used in fetched tweets
    """
    Hashtag_df = pd.DataFrame(columns=["Hashtag"])
    j = 0
    for tweet in range(0,len(results)):
        hashtag = results[tweet].entities.get('hashtags')
        for i in range(0,len(hashtag)):
            Htag = hashtag[i]['text'] 
            Hashtag_df.at[j,'Hashtag']=Htag
            j = j+1
    Hashtag_Combined = " ".join(Hashtag_df['Hashtag'].values.astype(str))
    text=" ".join(dataset['text'].values.astype(str))
    cleaned_text = " ".join([word for word in text.split()
                                    if word !="https"
                                    and word !="RT"
                                    and word !="co"
                                                   ])                         
    wc = WordCloud(width=500,height=500,background_color="white", stopwords=STOPWORDS).generate(Hashtag_Combined)
    plt.imshow(wc)
    plt.axis("off")
    r =random.randint(1,101)
    st = 'static\hashtag'+ str(r) +'.png'
    plt.savefig(st, dpi=300) 

    """
    Following piece of code is used to get a list of top 5 hashtags
    """
    hashtag=Hashtag_Combined.split(" ")
    df=pd.DataFrame()
    df['hashtags']=pd.Series([i for i in hashtag])
    data=df['hashtags'].value_counts()
    tag_count_list = data.values[:5]    
    tag_list = data.keys()[:5]

    """
    Following piece of code generates tokens using training set.
    """
    x=np.load('../Classification network/X_train.npy',allow_pickle=True)
    tk=Tokenizer(num_words=80000)
    tk.fit_on_texts(x)

    """
    Following piece of code is used to preprocess the fetched tweets. Preprocessing steps : 
    -> Remove links, hashtag symbols and twitter handles.
    -> Convert text to lowercase.
    -> Remove punctuations.
    -> Remove 'rt' from the text. 
    """
    sentDataFrame = dataset.copy(deep=True)
    sentDataFrame['text']=sentDataFrame['text'].apply(lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
    sentDataFrame["text"]=sentDataFrame["text"].apply(lambda x: x.lower())
    sentDataFrame["text"]=sentDataFrame["text"].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
    sentDataFrame["text"]=sentDataFrame["text"].apply(lambda x: x.replace('rt',''))

    """
    Following poece of code is used to generate vector of tokens for each tweet and padding the vectors such that every vector will have a length of 35.
    """
    tweet_tokens = tk.texts_to_sequences(dataset['text'].values)
    tweet_tokens_pad = pad_sequences(tweet_tokens, maxlen=35,padding='post')

    """
    Following poece of code is used to load the model for sentiment classification.
    """
    json_file = open("../Classification network/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../Classification network/model.h5")

    """
    Performing predictions using the model
    """
    senModelList=model.predict(x=tweet_tokens_pad)
    em = dataset["emojis"].values

    """
    Following piece of code stores sentiment score for each emoji in a dictionary 'dict'.
    """
    df_emojis = pd.read_csv('emoji_sentiment.csv')        
    x = df_emojis['emoji'].values
    y = df_emojis['sentiment_score'].values
    dict = {}
    for i in range(len(x)):
        dict[x[i]] = y[i]

    """
    Following piece of code performs averaging between the sentiment score obtained from the model and the sentiment score of the emojis present in the model. 
    """
    for q in range(len(em)):
        if(em[q]!='nan'):
            em_sent_score = 0
            sc_list = em[q].split(",")
            emj_count = 0
            for emj in sc_list:
                if emj in dict.keys():
                    em_sent_score = em_sent_score + dict[emj]
                    emj_count += 1
            if(emj_count>0):
                senModelList[q] = (((em_sent_score/emj_count)+1)/2 + senModelList[q])/2
    
    """
    Following piece of code classifies the sentiment of the tweet and stores it in the datafrane 'dataset'.
    """            
    senList = []
    for i in range(num):
        if(senModelList[i]<=0.5):
            senList.append('n')
        else:
            senList.append('p')
    dataset['sentiment'] = pd.Series(senList)
    
    """
    Following piece of code stores the sum of positively classified tweets in 'posSentPer' and negatively classified tweets in 'negSentPe' which are than stored in the list 'opList'.
    """
    posSentPer = len(dataset[dataset['sentiment']=='p'].sentiment)
    negSentPer = len(dataset[dataset['sentiment']=='n'].sentiment)
    opList = [posSentPer,negSentPer] 

    """
    Following piece of code stores the positive visibility score in 'posVis' and negative visibility score in 'negVis' which are than combinely stored in the list 'vbList'.
    """
    pos_dataset_for_visibility = dataset[dataset['sentiment']=='p'] 
    posVis = pos_dataset_for_visibility['followers'].sum(axis=0,skipna=True)
    neg_dataset_for_visibility = dataset[dataset['sentiment']=='n'] 
    negVis = neg_dataset_for_visibility['followers'].sum(axis=0,skipna=True)
    vbList = [posVis,negVis]

    """
    Following piece of code stores the tweets in 'tw_text', username of tweets in 'tw_uname' and number of followers of the authors of tweets in 'tw_foll'.
    """
    tw_uname = dataset['username'].values.tolist()
    tw_text = dataset['text'].values.tolist()
    tw_foll = dataset['followers'].values.tolist()

    return render_template('analysis.html',title='analysis',vbList=vbList,key=key,r=r,tag_list = tag_list,opList=opList,tag_count_list=tag_count_list,tw_uname=tw_uname,tw_text=tw_text,tw_foll=tw_foll)
    
app.run(host='localhost')
