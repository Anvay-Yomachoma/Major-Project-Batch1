import re
import tweepy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from langdetect import detect

# Set up Tweepy API keys
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with Tweepy API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define function to preprocess text
def preprocess_text(text, lang):
    if lang == 'en':
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\S+|#\S+', '', text)
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('english'))
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    elif lang == 'hi':
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\S+|#\S+', '', text)
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('hindi'))
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    elif lang == 'zh':
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\S+|#\S+', '', text)
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('chinese'))
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    elif lang == 'es':
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\S+|#\S+', '', text)
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('spanish'))
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    elif lang == 'it':
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\S+|#\S+', '', text)
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('italian'))
        words = nltk.word_tokenize(text)
       
