import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle


# Read Csv file
data = pd.read_csv("/home/codenomad/Desktop/Krish_NLP_ML/Dataset/Churn_Modelling.csv")

# Delete Unneccessary rows 
data = data.drop(["RowNumber", "CustomerId" , "Surname"], axis=1)

# change categorical values in numerical forms
label_encoder =LabelEncoder()
data["Gender"] =label_encoder.fit_transform(data["Gender"])

# Also change Geography columns 
one_hot_encoder =OneHotEncoder()
geo_one_hot_encoder = one_hot_encoder.fit_transform(data[["Geography"]])
geo_encode_df =pd.DataFrame(geo_one_hot_encoder.toarray() ,columns=one_hot_encoder.get_feature_names_out(["Geography"]))
data = pd.concat([data.drop("Geography", axis=1), geo_encode_df], axis=1)


# Split the data into features and target
X_feature =data.drop("EstimatedSalary", axis=1)
Y_feature = data["EstimatedSalary"]


# train testing data 
X_train , X_test , Y_train , Y_test = train_test_split(X_feature , Y_feature , random_state= 42)


# Applied Feature scaling 
scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# save lable encoder , one hot encoder and scaler into pickle files
with open("/home/codenomad/Desktop/Krish_NLP_ML/regression_pickle_files/label_encoder_gender.pkl", "wb") as file:
    pickle.dump(label_encoder , file)
    
with open("/home/codenomad/Desktop/Krish_NLP_ML/regression_pickle_files/onehot_encoder_geo.pkl", "wb") as file:
    pickle.dump(one_hot_encoder , file)
    
with open("/home/codenomad/Desktop/Krish_NLP_ML/regression_pickle_files/scaler.pkl", "wb") as file:
    pickle.dump(scaler , file)
 
 
######################   Train the Regression ANN Model    ##################################
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping , TensorBoard
import datetime


# Build the model
model =Sequential(
    [
        Dense(64 , activation ="relu" , input_shape=(X_train.shape[1] , )),
        Dense(32 , activation ="relu"),
        Dense(1)            # here dnn't need to pass activation function , there is default activation called "linear activation function"
    ]
)

# compile the model 
model.compile(optimizer ="adam", loss="mean_absolute_error", metrics=["mae"])

# set up log directory 
log_dir = "regressionlogs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callbacks = TensorBoard(log_dir =log_dir ,histogram_freq=1)

#### SET UP EARLY STOPPING
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience =10, restore_best_weights=True)

history =model.fit(
    X_train , Y_train , validation_data =(X_test , Y_test), epochs =100, callbacks =[early_stopping_callback ,tensorflow_callbacks ]
)

# Save the model.
model.save("/home/codenomad/Desktop/Krish_NLP_ML/regression_pickle_files/model.keras")