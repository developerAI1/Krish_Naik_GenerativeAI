"""
TF-IDF   is represent to -------> [Term Frequency - Inverse Document Frequency]

Dataset:
    sentence1 = good boy
    sentence2 = good girl
    sentence3 = boy girl good
    
Formulee of TF-IDF

    Term Frequency (TF) =   Number of repitation word in sentence
                            ________________________________________
                            Number of words  in senetence
                            
    Inverse Document Frequency  =   loge (Total Number of sentences)
                                    ________________________________________
                                    Number of sentences containing the word.

Working Rule;

        Term Frequency                                              IDF
    
        S1      S2      S3                                          words               IDF
    
good    1/2     1/2     1/3                                         good                loge(3/3)               loge(1) =0

boy     1/2     0       1/3                                         boy                 loge(3/2)

girl    0       1/2     1/3                                         girl                loge(3/2)




Now Make Actual Vectors based on the TF * IDF

                        good            boy                          girl
            sent1       1/2*0       1/2 *loge(3/2)                     0
            
            sent2       1/2 *0          0                              1/2 *loge(3/2)
            
            sent3       1/3 *0          1/3*loge(3/2)                  1/3 *loge(3/2)
            
Advantages and Disadvantages of TF-IDF

Advantages:

    1. Intutive
    2. Fixed length ===> vocab size
    3. word importance is capture
    
Disadvantages :

    1. sparsing still exist
    2. OOV   

"""


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

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
    review= [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review =" ".join(review)
    corpus.append(review)


# convert text into TF-IDf
tfidf =TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(corpus).toarray()
# print(tfidf.vocabulary_)

print()
print(X[0])