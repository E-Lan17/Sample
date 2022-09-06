# at the top of the file, before other imports
import warnings
warnings.filterwarnings('ignore')

# Importing the libraries
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Create a timer :       
start_time = datetime.now()
print('Data preprocessing...')

# Create a sub timer :       
sub_timer_start = datetime.now()

# Importing the dataset
dataset = pd.read_csv('Windows_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
feat_dataframe = dataset.iloc[:, :-1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    shuffle=True, random_state = 42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# End timer and display it :
sub_timer_finish = datetime.now()
print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))

# create a list of results
results = []

# Function for all the estimators:
def function_estimator(model, parameters) :
    print(model)
    pipeline = Pipeline([('Selected_features', SelectKBest()),
                         ('Estimator', model)])
    estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)
    # Train the model for the best parameters for the best number of features :
    estimator.fit(X_train, y_train)
    # Create a list for the results for this estimator :
    estimator_lst = []
    estimator_lst.append(model) 
    # Add the best parameters to the results :
    estimator_lst.append(str(estimator .best_params_))     
    # Add the name of the selected features to the results :
    estimator_lst.append('Selected features : ')
    best_feat = estimator.best_estimator_.named_steps['Selected_features'].get_support()
    best_feat = feat_dataframe.columns[best_feat].values
    estimator_lst.append(len(best_feat))
    estimator_lst.append(best_feat)
    # Add the results of the GridSearchCV :    
    estimator  = estimator.best_estimator_ 
    #Cross validation with new model and features on the new x_test :
    scores = cross_val_score(estimator , X_test, y_test,cv=5)                  
    # Add the result of the cross validation to the results :
    estimator_lst.append('Final cross_val_score on test data : ') 
    estimator_lst.append(str("%0.2f accuracy with a standard deviation of %0.2f"
                             % (scores.mean(), scores.std())))
    # Add the final results to a list of results :     
    results.append(estimator_lst)

# function for selectbest() because they are callable not string. but just use fclassif at the end:
def my_f_classif(X, y):
    return f_classif(X, y)

# Decision Tree Classifier
try :
    # Create a sub timer :       
    sub_timer_start = datetime.now()
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier() 
    parameters = {"Selected_features__score_func" : [my_f_classif],
                  "Selected_features__k": [9,10,11,12,13,14,15], # k: [1,2,3,4,5,6,7,8,9,10] or ["all"]
                  "Estimator__criterion": ['entropy','gini'],
                  "Estimator__splitter": ['best','random'],
                  "Estimator__random_state": [42]}
    function_estimator(model, parameters)
    # End timer and display it :
    sub_timer_finish = datetime.now()
    print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))
except :
    results.append('Decision Tree Classifier : ') 
    results.append('Error') 
    print('Error')
    pass 

# K-Nearest Neighbors (K-NN)
try :
    # Create a sub timer :       
    sub_timer_start = datetime.now()    
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier() 
    parameters = {"Selected_features__score_func" : [my_f_classif],
                  "Selected_features__k":[1,2,3,4,5,6,7,8,9,10], # k: int or ["all"]
                  "Estimator__n_neighbors": [1,2,3,4,5,6,7,8,9,10],
                  "Estimator__weights": ['uniform','distance'],
                  "Estimator__algorithm": ['ball_tree','kd_tree','brute'],
                  "Estimator__metric" : ['minkowski', 'chebyshev'],
                  "Estimator__p": [1,1.25,1.5,1.75,2]} 
    function_estimator(model, parameters)
    # End timer and display it :
    sub_timer_finish = datetime.now()
    print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))    
except :
    results.append('K-Nearest Neighbors : ') 
    results.append('Error') 
    print('Error')
    pass 

# Logistic Regression
try :
    # Create a sub timer :       
    sub_timer_start = datetime.now()    
    from sklearn.linear_model import LogisticRegression 
    model = LogisticRegression()
    parameters = {"Selected_features__score_func" : [my_f_classif],
                  "Selected_features__k":[1,2,3,4,5,6,7,8,9,10], # k: int or ["all"]
                  "Estimator__solver": ['newton-cg','lbfgs','liblinear','sag','saga'],
                  "Estimator__max_iter": [2000],
                  "Estimator__C": [0.1,0.125,0.15,0.175,2],
                  "Estimator__random_state": [42]} 
    function_estimator(model, parameters)
    # End timer and display it :
    sub_timer_finish = datetime.now()
    print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))    
except :
    results.append('Logistic Regression : ') 
    results.append('Error') 
    print('Error')
    pass 

# Naive Bayes
try :
    # Create a sub timer :       
    sub_timer_start = datetime.now()    
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    parameters = {"Selected_features__score_func" : [my_f_classif],
                  "Selected_features__k":[5,6,7,8,9,10], # k: int or ["all"]
                  }
    function_estimator(model, parameters)
    # End timer and display it :
    sub_timer_finish = datetime.now()
    print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))    
except :
    results.append('Naive Bayes : ') 
    results.append('Error') 
    print('Error')
    pass 

# Random Forest Classifier
try :
    # Create a sub timer :       
    sub_timer_start = datetime.now()    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    parameters = {"Selected_features__score_func" : [my_f_classif],
                  "Selected_features__k":[1,2,3,4,5,6,7,8,9,10], # k: int or ["all"]
                  "Estimator__n_estimators": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
                                   18,19,20,21,22,23,24,25,26,27,28,29,30],
                  "Estimator__criterion": ['entropy','gini'],
                  "Estimator__random_state": [42]}
    function_estimator(model, parameters)
    # End timer and display it :
    sub_timer_finish = datetime.now()
    print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))    
except :
    results.append('Random Forest Classifier : ') 
    results.append('Error') 
    print('Error')
    pass 

# Support Vector Machine
try :
    # Create a sub timer :       
    sub_timer_start = datetime.now()    
    from sklearn.svm import SVC
    model = SVC()
    parameters = {"Selected_features__score_func" : [my_f_classif],
                  "Selected_features__k":[5,6,7,8,9,10], # k: int or ["all"]
                  "Estimator__kernel": ['linear','poly','rbf'],
                  "Estimator__degree": [1,2,3,4,5],
                  "Estimator__gamma": ['scale','auto'],
                  "Estimator__random_state": [42]}
    function_estimator(model, parameters)
    # End timer and display it :
    sub_timer_finish = datetime.now()
    print('Done. Duration: {}'.format(sub_timer_finish - sub_timer_start))    
except :
    results.append('Support Vector Machine : ') 
    results.append('Error') 
    print('Error')
    pass 

# Create new file with a list .txt :
with open('Estimators_results.txt', 'w+') as filehandle:
    for listitem in results:
        filehandle.write('%s\n' % listitem)
print(results)

# End timer and display it :
end_time = datetime.now()
print('Total duration: {}'.format(end_time - start_time))