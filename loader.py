import pickle 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re 
import numpy as np 



STOPWORDS  =  pickle.load(open("models/stopwords.pkl","rb"))
rf =  pickle.load(open("models/rf_model.pkl","rb"))
nb  =  pickle.load(open("models/rf_model.pkl","rb"))
xgb =  pickle.load(open("models/rf_model.pkl","rb"))
tfidf =  pickle.load(open("models/tfidf.pkl","rb"))
#fasttext_model =  fasttext.load_model("models/fasttext_model.bin")
vader_model =  SentimentIntensityAnalyzer()

def extract_score(model,text):
    score = model.polarity_scores(text)

    if score['pos'] >=  score['neg']:
        sentiment = 1
    else:
        sentiment = 0 
    return sentiment



def sentiment_analyzer(review):
    sentiment= TextBlob(review)
    score= sentiment.sentiment.polarity
    if score >= 0:
        return 1
    elif score < 0:
        return 0 

def clean_text(text):
	text = str(text).lower()
	text = re.sub("[^\w\s]"," ",text)
	text =  ' '.join([w for w in text.split() if w not in STOPWORDS])
	return text




def analyze(text):
    text =  clean_text(text)
    text =  np.array(tfidf.transform([text]).todense())
    nb_pred =  nb.predict(text)[0]
    rf_pred =  rf.predict(text)[0]
    xgb_pred =  xgb.predict(text)[0]
    return (nb_pred,rf_pred,xgb_pred) 