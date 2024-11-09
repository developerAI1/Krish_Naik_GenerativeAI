# STEP -1  ( import modules and packages )
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle

# STEP -2 (Load Dataset )
data =pd.read_csv("/home/codenomad/Desktop/Krish_NLP_ML/Dataset/Churn_Modelling.csv")

# STEP -3 (preprocess the data )
# Drop irrelevant columns
data =data.drop(["RowNumber" , "CustomerId", "Surname"], axis=1)        # axis =1 means column wise




# STEP-4 (Encode categorical variable)
"""
Label Encoder is perfect for binary label, while using ANN, if we have only two values in categorical columns then 
we can use LabelEncoder
"""
lable_encoder_gender =LabelEncoder()
data["Gender"] =lable_encoder_gender.fit_transform(data["Gender"])



## STEP -5 (one hot encode 'Geography column")
"""
Label Encoder is not work well if categorical columns has more values from two specially ,while using ANN.
because ANN model is also able to do numerical calculations for example:

if column has three values spain , france and germany while we implement LabelEncoder then it return values as [0,1,2]
so it create confusion to model looks ANN cosider it as:

            2 greater  then 1 , 1 greater than 0 .
            
    to solve this problem we should need to use OneHotEncoder
"""
onehot_encoder_geo =OneHotEncoder()
geo_encoder=onehot_encoder_geo.fit_transform(data[["Geography"]])
" output should like [Geography_France , Geography_spain , Geography_germany] , means it create three diffrent columns"



# STEP -6 (get values of geo_encoder )
geo_encoded_df =pd.DataFrame(geo_encoder.toarray(), columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))


# STEP -7 (COMBIME ONE HOT ENCODING COLUMN WITH THE ORIGINAL DATA)
data =pd.concat([data.drop("Geography", axis=1), geo_encoded_df], axis=1)



# STEP -8  (Save the labelencoder and onehotencoder )
with open("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/label_encoder_gender.pkl", "wb") as file:
    pickle.dump(lable_encoder_gender , file)
    
with open("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/onehot_encoder_geo.pkl", "wb") as file:
    pickle.dump(onehot_encoder_geo , file)
 
    
# STEP -9
## Divide  the dataset into independent and dependent features
X = data.drop(["Exited"], axis=1)
Y = data["Exited"]
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.25 , random_state=42)



# STEP -10 (Scale these features)
"""
Understading the transformers(StandardScaler).
Mean of 0:
    After scaling, the average (mean) value of each feature (e.g., "Age" or "Salary") becomes 0. 
    This is done by subtracting the mean of the original feature values from each individual value.

Standard Deviation of 1:
    After scaling, each feature’s values have a standard deviation of 1. Standard deviation measures how spread out the values are from the mean. 
    By dividing by the standard deviation, we ensure that the spread or dispersion of each feature’s values is standardized.    
    
formula = Xscaled = (X - mean ) / standard deviation
where X is original value 
mean is average value of the feature.
Standard Deviation: A measure of the spread or variance in the feature values.
"""
scaler =StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# SAVE THE SCALER In Pickle file
with open("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/scaler.pkl", "wb") as file:
    pickle.dump(scaler , file)



# STEP -11 (ANN IMPLEMENTATION)
"""
Basic parameters to train ANN model

1. Sequential (N/W)
2. Dense  --> 32 , 64 128 ,,,
3. Activation function  ----> sigmoid , tanh , relu , leaku relu  (for multiple classification use 'Softmax')
4. Optimizer ---> Back Propogation  ---> (Updating the weights )
5. Loss function 
6. metrics ----> for classification ['accuracy'] and for regression [mse , mae]
7. training ---> Logs ------> Tensorboars (Display the graph for visualization)
"""


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import  EarlyStopping , TensorBoard
import datetime

##### Build ANN Model
model =Sequential(
    [
        Dense(64 , activation ="relu", input_shape = (X_train.shape[1], )),                 # First HL1 connected with input layer
        Dense(32 , activation ="relu") ,                                                    ## HL2,
        Dense(1, activation ="sigmoid")                                                     # output layer
        
    ]
    )
# Check the total parameter of model
total_parameter = model.summary()

# customly define optimizer and loss function
opt = tensorflow.keras.optimizers.Adam(learning_rate =0.01)
loss= tensorflow.keras.losses.BinaryCrossentropy()

### to check forward propagation  compile the model
model.compile(optimizer =opt, loss =loss , metrics =['accuracy'])           # if there is multiple classification feature then use loss = SparseCrossentropy


##### SET UP THE TENSORBOARD
log_dir = "logs/fit"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callbacks = TensorBoard(log_dir =log_dir ,histogram_freq=1)


#### SET UP EARLY STOPPING
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience =10, restore_best_weights=True)

## TRAIN THE MODEL 
history = model.fit(
    X_train , Y_train , validation_data = (X_test , Y_test) , epochs = 100 , 
    callbacks =[tensorflow_callbacks ,early_stopping_callback]
)

# Save the model.
model.save("/home/codenomad/Desktop/Krish_NLP_ML/pickle_file_directory/model.keras")

# Load Tensorboard Extension 
