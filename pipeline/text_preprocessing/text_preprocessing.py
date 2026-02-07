import pandas as pd
#define your data path here
data_path = "/home/mbithi002/Desktop/generativeai/data/IMDB Dataset.csv"

#load your data set
df = pd.read_csv(data_path)

#print(df.shape)

#print(df.head())

df = df.head(100)
#perform text preprocessing
#convert all text to lowercase
df['review'] = df["review"].str.lower()

#print(df["review"][3])





#remove html tags (only if necessary)

import re
def remove_html_tags(text: str) -> str:
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

df['review'] = df['review'].apply(remove_html_tags)
#print(df["review"])





#remove urls (only if necessary)

def remove_urls(text: str) -> str:
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)

df['review'] = df['review'].apply(remove_urls)
# print(df["review"])
# print(remove_urls("\n Check out this link: http://example.com"))





# Handle punctuation and special characters

import string
def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

df['review'] = df['review'].apply(remove_punctuation)
# print(remove_punctuation("Hello, world! This@# is a test."))
# print(df["review"])
# print(remove_punctuation(df["review"][5]))





# handle abbreviations and contractions

import contractions
def expand_contractions(text: str) -> str:
    return contractions.fix(text)

# print(expand_contractions("I can't do this."))

chat_words = {
    'AFAIK':'As Far As I Know',
    'AFK':'Away From Keyboard',
    'ASAP':'As Soon As Possible',
    "FYI": "For Your Information",
    "ASAP": "As Soon As Possible",
    "BRB": "Be Right Back",
    "BTW": "By The Way",
    "OMG": "Oh My God",
    "IMO": "In My Opinion",
    "LOL": "Laugh Out Loud",
    "TTYL": "Talk To You Later",
    "GTG": "Got To Go",
    "TTYT": "Talk To You Tomorrow",
    "IDK": "I Don't Know",
    "TMI": "Too Much Information",
    "IMHO": "In My Humble Opinion",
    "ICYMI": "In Case You Missed It",
    "AFAIK": "As Far As I Know",
    "BTW": "By The Way",
    "FAQ": "Frequently Asked Questions",
    "TGIF": "Thank God It's Friday",
    "FYA": "For Your Action",
    "ICYMI": "In Case You Missed It",
}

def expand_chat_words(text: str) -> str:
    for word, expansion in chat_words.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', expansion, text)
    return text

# print(expand_chat_words("FYI, I'll be AFK for a while."))

# while handling abbreviations, you can choose from available packages e.g abbreviations-py, abbrfix etc but here context and intent matters when choosing the right package. For example, if you are working with medical text, you might want to use a package that is specifically designed for medical abbreviations. If you are working with general text, you might want to use a package that is more general-purpose.





# incorrect spelling handling

from textblob import TextBlob
incorrect_text = "I havv a speling eror."
def correct_spelling(text: str) -> str:
    return str(TextBlob(text).correct())
# print(correct_spelling(incorrect_text))




# handling stop words
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# print(stop_words)
# print(len(stop_words))
def remove_stop_words(text: str) -> str:
    return ' '.join([word for word in text.split() if word not in stop_words]) 

# print("\n ")
# print(df["review"][5])
# print("\n ")
# df["review"] = df["review"].apply(remove_stop_words)
# print(df["review"][5])




# Handl;ing emojis, how ever you can choose to keep emojis if they are relevant to your analysis, for example, in sentiment analysis, emojis can provide valuable information about the sentiment of the text. In such cases, you might want to use a package that can handle emojis, such as emoji or emot.
import emoji
emoji.demojize("Loved the movie. It was 😘😘")
emoji.emojize("Loved the movie. It was :face_blowing_a_kiss::face_blowing_a_kiss:")




# Tokenization, can ber done using split method, regular expressions or using packages

# using nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt_tab')
text = "Hello world! This is a test."
# print(word_tokenize(text))
# print(sent_tokenize(text))

# usiong spacy
import spacy
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(text)





# Stemming and Lemmatization, can be done using packages such as nltk, spacy, snowballstemmer etc. Stemming is the process of reducing a word to its base form, while lemmatization is the process of reducing a word to its base form while considering the context and meaning of the word. For example, the word "running" would be reduced to "run" in stemming, but in lemmatization, it would be reduced to "run" if it is used as a verb and "running" if it is used as a noun.

# using nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


sample_text = "player plays, playing, played"
print([ps.stem(word) for word in sample_text.split()])


# lemmatization using nltk
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
wordnet_lemmatizer = WordNetLemmatizer()

import nltk
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos='v')))


# NOTEBOOK http://colab.research.google.com/drive/1AOME-s54Jh8HxGf4i2hTHSFU7jkkfQLY#scrollTo=eAf31FTGfXJe