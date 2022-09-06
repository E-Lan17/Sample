# K-Means Clustering

# Importing the libraries
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans

# Create a timer :       
start_time = datetime.now()

# Importing the Cluster_data.csv
dataset = pd.read_csv('Cluster_data.csv')
# print(dataset)

## Frequency encoding :
    
# size of each category (number of time each activity appears) :
encoding = dataset.groupby('Activities').size()
# print(encoding)

# get frequency of each category, add it to dataframe and replace the order of index :
encoding = encoding/len(dataset)
dataset['Encode'] = dataset.Activities.map(encoding)
dataset = dataset[["Duration", "Activities", "Encode", "Day"]]
#print(dataset)

# Select the column the dataset
X = dataset.iloc[:,[2, 3]].values


# Number of cluster :
N = 111

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = N, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#Add the cluster to the dataset and display the activities by cluster :
cluster_map = dataset
cluster_map['cluster'] = y_kmeans
df_cluster = pd.DataFrame()

for i in range(N) :
    df_cluster = df_cluster.append(cluster_map[cluster_map.cluster == i])

# Create a dictionary with the sorted activities with value 0 for each key :
dict_cluster = { i : None for i in range(N) }
for i in range(len(df_cluster)) :
        if dict_cluster[df_cluster.iloc[i,4]] == None :
            dict_cluster[df_cluster.iloc[i,4]] = df_cluster.iloc[i,1]
        if df_cluster.iloc[i,1] not in dict_cluster[df_cluster.iloc[i,4]] :  
            dict_cluster[df_cluster.iloc[i,4]] = dict_cluster[df_cluster.iloc[i,4]] +" " + df_cluster.iloc[i,1]
                        
# Create new file with a dict .txt (change dict name):   
with open("Cluster.txt", 'w') as filehandle:  
    for key, value in dict_cluster.items():  
        filehandle.write('%s:%s\n' % (key, value))
 
# End timer and display it :
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
        
