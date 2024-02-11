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
Y = news_dataset['label']



#Stemming 

stem = PorterStemmer()


def stemming(text):
    stem_text = re.sub('[^a-zA-Z]', ' ', text)
    stem_text = stem_text.lower()
    stem_text = stem_text.split()
    stem_text = [stem.stem(word) for word in stem_text if not word in stopwords.words('english')]
    stem_text = ' '.join(stem_text)    
    return stem_text


news_dataset['text'] = news_dataset['text'].apply(stemming)

x = news_dataset['text'].values


# print('hello there')


# print(x.head())
y = news_dataset['label'].values


#Converting textual data into numeric data


vect = TfidfVectorizer()
vect.fit(x)

X = vect.transform(x)


# Spliting into X and Y

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.2, random_state=2)


#Model Training

model = LogisticRegression()

model.fit(Xtrain, Ytrain)

# Evaluation

xtrain = model.predict(Xtrain)
print(xtrain)


Acc = accuracy_score(xtrain, Ytrain)
print(Acc)


# Evaluation on test data

xtest = model.predict(Xtest)

print(xtest)

Acc = accuracy_score(xtest, Ytest)

print(Acc)


# let's test out the model with real time data

X_news = Xtest[17]
predict = model.predict(X_news)

print(predict)

if predict[0] == 'REAL':
    print('the news is real')
else:
    print('the news is fake')







