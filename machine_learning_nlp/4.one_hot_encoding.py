"""
One hot encoding : -
The one hot encoding is basically used for conver text into vector.
Working of One Hot Encoding.
 
For  Example:
    sentences : 
        1. The food is good.
        2. The food is bad.
        3. pizza is amazing.
        
    
vocabulary = is collection of unique words , so the vocab for above 3 sentences is:
    vocab =[The , food , is , good , bad , pizza , amazing]
    
    here size of vocab is total words exist in list =  7
    
After implement OHE:

vectors: 
        sentence1  = [
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0]
            [0,0,0,1,0,0,0]
        ]  
            shape = 4 x 7
        
        sentence 2 = [
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0]
        ]   
            shape = 4 x 7
            
        sentence 3 = [
            [0,0,0,0,0,1,0],
            [0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1]
        ]
        
            shape = 3 x 7
        
** Important **
    - while we train the machine learning model then does not use 'ONE HOT ENCODING"

Advantages:
    1. Easy to implement with python 

            libraries : [sklearn , OneHotEncoder, pd.get_dummies()]

Disadvantages:
    1. sparse matrix = overfitting (very much zeros)
    
    2 . Ml algorithims need to fixed size of all features but in above sentence third one have 3 x7 lenght , so we can not trained model with using one hot encoding.
    
    3. no semantic meaning capture.
    
    4. Out of vocabulary.
    
        means if predict test sentence : "Burger is good food"
        
            here 'burger' not  in  vocab =[The , food , is , good , bad , pizza , amazing]
                so it increase problem for new word.
    """
    
