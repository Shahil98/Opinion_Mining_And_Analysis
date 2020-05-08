# Opinion Mining And Analysis
This is a NLP, Deep Learning and Data Analysis Project. 
<br>
## Types of analysis performed in this project on tweets fetched using a keyword or sentence searched
1) Displaying most frequently used hashtags.
2) Sentiment Analysis
3) Visibility Analysis i.e. calculating the reach of positive tweets vs negative tweets.
## Sentiment Classification Architecture
(https://github.com/Shahil98/Opinion_Mining_And_Analysis/blob/master/Architecture_Sentiment_Classification.jpg?raw=true)
# Getting started
1) Clone this repository.
```
git clone https://github.com/Shahil98/Opinion_Mining_And_Analysis.git
```
2) Create a twitter developer account.
3) Execute ```pip install -r requirements.txt``` to install necessary libraries.
4) Download dataset from ```https://www.kaggle.com/kazanova/sentiment140``` and place the csv file inside Classification network folder.
5) Execute ```python generate_data.py``` inside Classification network folder to generate data for training the sentiment classification network.
6) Execute ```python network.py``` to train the netwotk. After execution is completed a model.h5 and model.json file will be created.
7) Move to webApp folder and execute ```set FLASK_APP=main.py```.
8) Add twitter developer credentials in main.py file.
9) Execute ```python main.py``` to start the application at ```http://localhost:5000```.
