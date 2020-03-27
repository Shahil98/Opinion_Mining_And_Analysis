from tensorflow.python.keras.models import model_from_json
from tensorflow.python.framework import ops
import tweepy
from twitter import *
def init():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model.h5")
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    graph = ops.get_default_graph()
    #Setting up keys to access twitter data
    consumer_key = '9lMcBOT8L4NIVaZGPjGYd5Hpw'
    consumer_secret = 'n6zdHxzQf9IoZRAoqjplpRvZ66poE7itfw4OhQWCJXBZfzK7Ki'
    access_token = '4866812614-UdFFNvUp1CnV0tAT3WyWSJdlaLLrjiHVJ5y7w0f'
    access_token_secret = 'Wrfm0zbzIIQCH3aCFTiIq38oi8TYcZNgWkaCTWQ7J5M06'
    #Interacting with twitter's API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    auth.set_access_token(access_token, access_token_secret)
    #creating the API object
    api = tweepy.API (auth)

    twitter = Twitter(auth=OAuth('4866812614-UdFFNvUp1CnV0tAT3WyWSJdlaLLrjiHVJ5y7w0f',
                  'Wrfm0zbzIIQCH3aCFTiIq38oi8TYcZNgWkaCTWQ7J5M06',
                  '9lMcBOT8L4NIVaZGPjGYd5Hpw',
                  'n6zdHxzQf9IoZRAoqjplpRvZ66poE7itfw4OhQWCJXBZfzK7Ki'))

    return loaded_model,graph,api,twitter