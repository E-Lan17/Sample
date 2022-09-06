# Artificial Neural Network

# at the top of the file, before other imports
import warnings
warnings.filterwarnings('ignore')

# Importing the libraries :
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold as KFold

# Create a timer :       
start_time = datetime.now()

# Importing the dataset :
dataset = pd.read_csv('Windows_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
feat_dataframe = dataset.iloc[:, :-1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    shuffle=True, random_state = 42)
# Feature Scaling :
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Difining the neuronal model :
def create_model(neuron_input=40,
                 neuron_layer=4,
                 neuron_output=1,
                 optimizer="RMSprop",
                 loss="binary_crossentropy") :
    # Initializing the ANN
    model = tf.keras.models.Sequential()
    # Adding the input layer 
    model.add(tf.keras.layers.Dense(neuron_input, activation='relu'))
    # Adding the second hidden layer
    model.add(tf.keras.layers.Dense(neuron_layer, activation='relu'))
    # Adding the output layer
    model.add(tf.keras.layers.Dense(neuron_output, activation='sigmoid'))
    # Compiling the ANN
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    return model

# Wrap the model using the neuronal model :
clf = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model,verbose=0)

# Difining the model for selected features :
slc = SelectKBest()

# Tunning parameters :
parameters = {
          "Selected_features__k":["all"], # k: [1,2,3,4,5,6,7,8,9,10] or ["all"]
          "clf__neuron_input": [40], # [10,20,30,40,50,75,100]
          "clf__neuron_layer": [4], # [2,4,6]
          "clf__neuron_output": [1],
          "clf__optimizer": ["RMSprop"], # ["Adam","Adagrad","Adadelta","RMSprop"]
          "clf__loss": ["binary_crossentropy"], # ["binary_crossentropy","KLDivergence"]
          "clf__epochs": [300], # [200,250,300]
          "clf__batch_size" : [50] # [20,32,40]
          } 

# Difining the pipeline :
pipeline = Pipeline([
    ('Selected_features', slc),
    ('clf', clf)
    ]) 

# Difining the GridSearchCV and fit the model :
estimator = GridSearchCV(pipeline, parameters, refit=True, cv=5)                  
estimator.fit(X_train, y_train)

# Add the name of the selected features to the results :
best_feat = estimator.best_estimator_.named_steps['Selected_features'].get_support()
best_feat = feat_dataframe.columns[best_feat].values
results = []
results.append(best_feat)
results.append(estimator .best_params_)
print(best_feat)           
print(estimator .best_params_)

# Splitting data in KFold and predict on the test data:
accuracy = []
kf = KFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_test,y_test):
    X_f = X_test[test_index]
    y_f = y_test[test_index] 
    y_pred = estimator.predict(X_f)
    y_pred = (y_pred > 0.5)
    accuracy.append(float(accuracy_score(y_f, y_pred)))
accuracy = np.array(accuracy)
std = accuracy.std()
accuracy = accuracy.mean()

# Display the result :
results.append("%0.2f accuracy with a standard deviation of %0.2f "
                             % (accuracy,std))   
print(str("%0.2f accuracy with a standard deviation of %0.2f "
                             % (accuracy,std)))

# Create new file with a list .txt :
with open('ANN_results.txt', 'w+') as filehandle:
    for listitem in results:
        filehandle.write('%s\n' % listitem)
print(results)

# End timer and display it :
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

