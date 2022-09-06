# Importing the libraries
import pandas as pd
from datetime import date
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

## Preprocessing :
# Create a timer and get the date :       
Start_time = datetime.now()
Current_Date = str(date.today())

# Importing the dataset
dataset = pd.read_csv('SPLIT_SEQ_10activities.csv')
corpus = dataset.iloc[1:,1].values

# Creating the Bag of Words model
min_df = 0.0001 #0.0001 to 0.00005 ;0.00001 crash kernel
cv = CountVectorizer(min_df= min_df) 
X = cv.fit_transform(corpus)
X = X.toarray()
y = dataset.iloc[1:, -1].values
max_features = len(X[0])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# End sub timer :
Current_time = datetime.now()
# Partial results update
Partial_Task_Result = []
Partial_Task_Result.append('Data preprocessing :')
Partial_Task_Result.append('    min_df : %0.6f (%0.6f percent)' % (min_df,(min_df*100)))
Partial_Task_Result.append('    Token max features : %0.2f ' % max_features) 
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Start_time)) 
# Create a file with date and partial results update :
file_name = Current_Date+"_Partial_Task_Result.txt"
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

## Models : 
# Decision Tree Classifier
Partial_Task_Result = []
Partial_Task_Result.append('Decision Tree Classifier : ')
try:
    # Training the Decision Tree Classifier model on the Training set for the best parameters :
    # Create a sub timer :       
    Task_time_start = datetime.now()
    # Create the pieline with the desired parameters for the estimator :
    from sklearn.tree import DecisionTreeClassifier
    parameters = {"Selected_features__k": ["all"], # k: [1,2,3,4,5,6,7,8,9,10] or ["all"]
                      "Estimator__criterion": ['gini'], # ['entropy','gini']
                      "Estimator__splitter": ['random'], # ['best','random']
                      "Estimator__random_state": [42]}
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                     ('Estimator', DecisionTreeClassifier())])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Add the best parameters to the results :
    Partial_Task_Result.append('    %s' % estimator .best_params_)    
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    Partial_Task_Result.append('    Final cross_val_score on test data : ' 
                               '%0.2f accuracy with a standard deviation of %0.3f'
                             % (scores.mean(), scores.std()))
except :
    Partial_Task_Result.append('    Error') 
    pass 
# End timer and display it :
Current_time = datetime.now()
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Task_time_start))      
# Create new file with a list .txt (change list name):
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

# K-Nearest Neighbors (K-NN)
Partial_Task_Result = []
Partial_Task_Result.append('K-Nearest Neighbors (K-NN) : ')
try:
    # Training the K-Nearest Neighbors model on the Training set for the best parameters :
    # Create a sub timer :       
    Task_time_start = datetime.now()
    # Create the pieline with the desired parameters for the estimator :
    from sklearn.neighbors import KNeighborsClassifier
    parameters = {"Selected_features__k":["all"], # k: [1,2,3,4,5,6,7,8,9,10] or ["all"]
              "Estimator__n_neighbors": [50],
              "Estimator__weights": ['uniform'], #['uniform','distance']
              "Estimator__algorithm": ['kd_tree'], #['ball_tree','kd_tree','brute']
              "Estimator__metric" : ['minkowski'], #['minkowski', 'chebyshev']
              "Estimator__p": [1.5]} #[1,1.25,1.5,1.75,2]            
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                     ('Estimator', KNeighborsClassifier())])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Add the best parameters to the results :
    Partial_Task_Result.append('    %s' % estimator .best_params_)    
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    Partial_Task_Result.append('    Final cross_val_score on test data : ' 
                               '%0.2f accuracy with a standard deviation of %0.3f'
                             % (scores.mean(), scores.std()))
except :
    Partial_Task_Result.append('    Error') 
    pass 
# End timer and display it :
Current_time = datetime.now()
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Task_time_start))      
# Create new file with a list .txt (change list name):
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

# Logistic Regression
Partial_Task_Result = []
Partial_Task_Result.append('Logistic Regression : ')
try:
    # Training the Logistic Regression model on the Training set for the best parameters :
    # Create a sub timer :       
    Task_time_start = datetime.now()
    # Create the pieline with the desired parameters for the estimator :
    from sklearn.linear_model import LogisticRegression    
    parameters = {"Selected_features__k":["all"], # k: [1,2,3,4,5,6,7,8,9,10] or ["all"]
                  "Estimator__solver": ['lbfgs'], #['newton-cg','lbfgs','liblinear','sag','saga']
                  "Estimator__max_iter": [2000],# [1000,1500,2000]
                  "Estimator__C": [0.15], #[0.1,0.125,0.15,0.175,2]
                  "Estimator__random_state": [42]}
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                     ('Estimator', LogisticRegression())])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Add the best parameters to the results :
    Partial_Task_Result.append('    %s' % estimator .best_params_)    
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    Partial_Task_Result.append('    Final cross_val_score on test data : ' 
                               '%0.2f accuracy with a standard deviation of %0.3f'
                             % (scores.mean(), scores.std()))
except :
    Partial_Task_Result.append('    Error') 
    pass 
# End timer and display it :
Current_time = datetime.now()
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Task_time_start))      
# Create new file with a list .txt (change list name):
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

# Naive Bayes
Partial_Task_Result = []
Partial_Task_Result.append('Naive Bayes : ')
try:
    # Training the Naive Bayes model on the Training set for the best parameters :
    # Create a sub timer :       
    Task_time_start = datetime.now()
    # Create the pieline with the desired parameters for the estimator :
    from sklearn.naive_bayes import GaussianNB
    parameters = {}
    estimator = GaussianNB()
    parameters = {"Selected_features__k":["all"], # k: int or ["all"]
                  }
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                     ('Estimator', GaussianNB())])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Add the best parameters to the results :
    Partial_Task_Result.append('    %s' % estimator .best_params_)    
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    Partial_Task_Result.append('    Final cross_val_score on test data : ' 
                               '%0.2f accuracy with a standard deviation of %0.3f'
                             % (scores.mean(), scores.std()))
except :
    Partial_Task_Result.append('    Error') 
    pass 
# End timer and display it :
Current_time = datetime.now()
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Task_time_start))      
# Create new file with a list .txt (change list name):
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

# Random Forest Classifier
Partial_Task_Result = []
Partial_Task_Result.append('Random Forest Classifier : ')
try:
    # Training the Random Forest Classifier model on the Training set for the best parameters :
    # Create a sub timer :       
    Task_time_start = datetime.now()
    # Create the pieline with the desired parameters for the estimator :
    from sklearn.ensemble import RandomForestClassifier
    parameters = {"Selected_features__k":["all"], # k: [1,2,3,4,5,6,7,8,9,10] or ["all"]
                  "Estimator__n_estimators": [max_features], #[5,10,15,20,25,30]
                  "Estimator__criterion": ['gini'], #['entropy','gini']
                  "Estimator__random_state": [42]}
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                     ('Estimator', RandomForestClassifier())])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Add the best parameters to the results :
    Partial_Task_Result.append('    %s' % estimator .best_params_)    
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    Partial_Task_Result.append('    Final cross_val_score on test data : ' 
                               '%0.2f accuracy with a standard deviation of %0.3f'
                             % (scores.mean(), scores.std()))
except :
    Partial_Task_Result.append('    Error') 
    pass 
# End timer and display it :
Current_time = datetime.now()
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Task_time_start))      
# Create new file with a list .txt (change list name):
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

# Support Vector Machine
Partial_Task_Result = []
Partial_Task_Result.append('Support Vector Machine : ')
try:
    # Training the Random Forest Classifier model on the Training set for the best parameters :
    # Create a sub timer :       
    Task_time_start = datetime.now()
    # Create the pieline with the desired parameters for the estimator :
    from sklearn.svm import SVC
    parameters = {"Selected_features__k":["all"], # k: int or ["all"]
                  "Estimator__kernel": ['rbf'], #['linear','poly','rbf']
                  "Estimator__degree": [2],#[1,2,3,4,5]
                  "Estimator__gamma": ['auto'], #['scale','auto']
                  "Estimator__random_state": [42]}
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                     ('Estimator', SVC())])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Add the best parameters to the results :
    Partial_Task_Result.append('    %s' % estimator .best_params_)    
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    Partial_Task_Result.append('    Final cross_val_score on test data : ' 
                               '%0.2f accuracy with a standard deviation of %0.3f'
                             % (scores.mean(), scores.std()))
except :
    Partial_Task_Result.append('    Error') 
    pass 
# End timer and display it :
Current_time = datetime.now()
Partial_Task_Result.append('    Duration: {} \n'.format(Current_time - Task_time_start))      
# Create new file with a list .txt (change list name):
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)

# End total timer :
Current_time = datetime.now()
Partial_Task_Result = []
Partial_Task_Result.append('Total Duration : {} \n'.format(Current_time - Start_time)) 
# Create a file with Partial results update :
with open(file_name , 'a') as filehandle:
    for listitem in Partial_Task_Result :
        filehandle.write('%s\n' % listitem)
filehandle.close()
print(Partial_Task_Result)