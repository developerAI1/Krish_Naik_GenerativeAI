import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN , Dense
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping



# Load the imbd dataset
max_features = 10000
(X_train , Y_train) , (X_test , Y_test) =imdb.load_data(num_words =max_features)

##### Mapping of words index back to words (for understanding)
# word_index = imdb.get_word_index()
# print(word_index)


# word embedding representation
max_length =500
X_train =sequence.pad_sequences(X_train , maxlen =max_length)
X_test =sequence.pad_sequences(X_test , maxlen =max_length)


# Train Simple RNN
dimension = 128
model =Sequential()
model.add(Embedding(max_features, dimension , input_length = max_length))                   # Embedding Layers
model.add(SimpleRNN(dimension , activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Create an instance of early stopping callback
earlystopping =EarlyStopping(monitor ="val_loss", patience= 3, restore_best_weights =True)

# Train the model with early stopping 
history =model.fit(
    X_train , Y_train , epochs = 10 , batch_size = 32,validation_split =0.2, callbacks=[earlystopping]
)

model.save("/home/codenomad/Desktop/Krish_NLP_ML/Simple_RNN_Model/simple_rnn.keras")

