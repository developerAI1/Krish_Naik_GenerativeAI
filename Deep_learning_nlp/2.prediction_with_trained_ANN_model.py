import tensorflow 
from tensorflow.keras.models import load_model
import keras
import pandas as pd
import numpy as np
import pickle

# STEP - 1
### LOAD THE ANN TRAINED MODEL
loaded_model =load_model("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/model.keras")

# STEP - 2
### Load the encoder and scaler model
with open("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo= pickle.load(file)
    
with open("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender= pickle.load(file)
    
with open("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/scaler.pkl", "rb") as file:
    scaler= pickle.load(file)


# Example Input data
input_data = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender":"Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts":2,
    "HasCrCard":1,
    "IsActiveMember":1,
    "EstimatedSalary":50000
}


## convert into dataframe
dataframe =pd.DataFrame([input_data])
geo_encoder=onehot_encoder_geo.transform(dataframe[["Geography"]])
geo_encoded_df =pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

### Combine the one - hot encoding columns with the input data
dataframe =pd.concat([dataframe.drop("Geography", axis =1), geo_encoded_df], axis =1)

# also apply one hot encoding on Gender data
dataframe["Gender"] =label_encoder_gender.transform(dataframe["Gender"])


# Sacling the input data
input_scaled = scaler.transform(dataframe)


# make prediction 
prediction =loaded_model.predict(input_scaled)

####### here is binary classification 

if prediction > 0.5:
    print("The customer is likely to churn. ")
else:
    print("The customer is not likely to churn. ")





