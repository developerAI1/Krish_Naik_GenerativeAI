"""
Dataset :

    Text                                                    o/p
1. He is a good boy.                                        1                
2. she is a good girl.                                      1
3. Boy and girl are good.                                   1 
    
Steps to analyse vector dataset.
        
    Text Preprocessing.

        1. Lowercase text (to convert all text of dataset in lower case.)
        2. stopwords (use to remove unneccessary words from text dataset.)
        
    Vocabulary and frequency:

            Vocabulary                                 Frequency
    good (repeated 3 time in sentences)           =         3
    girl (repeated 2 time in sentences)           =         2
    boy (repeated 2 time in sentences)            =         2


    vocab shouold be created on frequency based.

    vocab =[good , girl , boy]

    vectorization :

        vector of 1 sentence = [1,0,1]
        vector of 2 sentence = [1,1,0]
        vector of 3 sentence = [1,1,1]
        
Difference between (binart BOW and Normal BOW)

binary BOW = {1, 0}
normal BOW = {count will get updated based on the frequency.}


Advantages and Disavantage of BOW.

    1. Simple and intutive
    2 . problem of fixed vector lenght of OHE solve in BOW.
    
Disadvantage:
    1. sparse matrix or array -> (overfitting)
    2. ordering of the word is getting change.
    3. out of vocabulary (OOV)
"""
# import libraries and modules

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# read the data
data =pd.read_csv("/home/codenomad/Desktop/Krish_NLP_ML/Dataset/SMSSpamCollection.tsv", sep="\t", names =["label", "messages"])

# get all message from dataset
messages=data["messages"]

# data cleaning and preprocessing
corpus =[]  
for i in range(1 , len(messages)):
    # keep only lowercase and uppercase text. replace every another thing with empty space
    review = re.sub('[^a-zA-Z]', " ",messages[i])
    review = review.lower()
    review = review.split()
    # Apply stemmer and stopwords to the text
    review= [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review =" ".join(review)
    corpus.append(review)
    
    
# convert text into bag of words
vectorizer =CountVectorizer(max_features=100, binary=True)              # pick the top 25 hundred frequency from the all dataset
X = vectorizer.fit_transform(corpus).toarray()
print(X)