from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np
# Sentences
sent =[
    "the glass of milk",
    "the glass of juice",
    "the cup of tea",
    "I am a good boy",
    "I am a good developer",
    "Understand the meaning of words",
    "your videos are good"
]


# Define the vocabulary Size 
voc_size =10000


# one hot representation for ever words
one_hot_repr =[one_hot(words , voc_size) for words in sent]


# word embedding representation
sent_length =8
embedding_docs = pad_sequences(one_hot_repr, padding="pre", maxlen = sent_length)


# feature representation
dim =10

model =Sequential()
model.add(Embedding(voc_size , dim, input_length= sent_length))
model.compile(optimizer="adam", loss="mse")

res =model.predict(embedding_docs)
print(res)