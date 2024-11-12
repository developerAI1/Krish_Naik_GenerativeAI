"""
Determine the optimal number of hidden layers and neurons for an Artifical Neural Network

This can be challenging and often requires experimentation. However , there are some guidelines and methods that can help you in making an
informed decission :
    
    a). Start Simple: Begin with a simple architecture and gradually increase complexity if needed
    b). Grid Search/ Random Search : Use grid search  or random search to try differnt architectures.
    c). cross-validation : Use cross -validation to evaluate the performance of differnt architectures.    
    d). Heuristics and Rules of Thumb : Some heuristics and empricial rules can provide starting points such as :
        1. The number of neuron in the hidden layer should be between the size of the input layer and the size of the output layer.
        2. A common practise is to start with 1-2 hidden layers.
    
"""


import pandas as pd
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
from sklearn.pipeline import Pipeline
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier



# read csv file:
data = pd.read_csv("/home/codenomad/Desktop/Krish_NLP_ML/Dataset/Churn_Modelling.csv")

# Drop irrelevant columns
data =data.drop(["RowNumber" , "CustomerId", "Surname"], axis=1)        # axis =1 means column wise


lable_encoder_gender =LabelEncoder()
data["Gender"] =lable_encoder_gender.fit_transform(data["Gender"])


onehot_encoder_geo =OneHotEncoder()
geo_encoder=onehot_encoder_geo.fit_transform(data[["Geography"]])

# STEP -6 (get values of geo_encoder )
geo_encoded_df =pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))


# STEP -7 (COMBIME ONE HOT ENCODING COLUMN WITH THE ORIGINAL DATA)
data =pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)


## Divide  the dataset into independent and dependent features
X = data.drop(["Exited"], axis=1)
Y = data["Exited"]
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.25 , random_state=42)


scaler =StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



#########           Define a function to create the model and try different parameters(KerasClassifier)    ##########################

def create_model(neurons=32,layers=1):
    model=Sequential()
    model.add(Dense(neurons,activation='relu',input_shape=(X_train.shape[1],)))
    for _ in range(layers-1):
        model.add(Dense(neurons,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
    return model


# Create KerasClassifier and define grid parameters
model=KerasClassifier(layers=1,neurons=32,build_fn=create_model,verbose=1)

# Define the grid search parameters
param_grid = {
    'neurons': [16, 32, 64, 128],
    'layers': [1, 2],
    'epochs': [50, 100]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3,verbose=1)
grid_result = grid.fit(X_train, Y_train)

# Print the best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



