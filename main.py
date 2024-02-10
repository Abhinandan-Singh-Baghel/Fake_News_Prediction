import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import nltk
nltk.download('stopwords')


# print(stopwords.words('english'))

# loading the dataset to a pandas DataFrame

news_dataset = pd.read_csv('news.csv')

print(news_dataset.shape)


print(news_dataset.head())

#checking for missing values
print(news_dataset.isnull().sum())

#seperating data and labels

X = news_dataset.drop(columns = 'label')
Y = df['label']



Stemming 

stem = PorterStemmer()


def stemming(text):
    stem_text = re.sub('[^a-zA-Z]', ' ', text)
    stem_text = stem_text.lower()
    stem_text = stem_text.split()
    stem_text = [stem.stem(word) for word in stem_text if not word in stopwords.words('english')]






