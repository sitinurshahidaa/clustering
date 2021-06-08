#First of all, import all libraries
import numpy as np 
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import pickle

model = DBSCAN()


#Load Dataset
data = pd.read_csv('FHData.csv')


print ("Shape of the DataFrame: ", data.shape)
print(data.head())

print("\n Dataset Info: \n")
print(data.info())

#Drop the single null value from On_LLT column of the dataset and convert it to int
data = data.dropna(subset=['On_LLT'])
data['On_LLT'] = data['On_LLT'].astype(int)
#Verify that null values are dropped
print("\n Dataset Info: \n")
print(data.info())

print("\n Print out the frequency of different target labels: \n")
print(data['DLCC Results'].value_counts())

#Hence we can see that its a case on imbalanced classes with "Unlikely" being way more frequent than other output labels. 
#Therefore our DBSCAN clustering should ideally cluster most of the data points into one large cluster and other data points into other distinct but 
#smaller clusters.

#Drop Target Label Column 
data=data.drop(columns='DLCC Results')

#Convert categorical features using one-hot encoding. This is required because DBSCAN does not work with categorical data.

categorical_columns  = ['Ethnic', 'Personal_PCAD', 'Gender', 'Personal_CVA_PVD', 'Smoking','Diabetes', 
'Family_Hypercholesterolaemia','Family_PCAD', 'Family_FH', 'On_LLT','Tendon_xanthomata', 'Corneal_arcus']

df = pd.DataFrame()
for col in categorical_columns:
    y = pd.get_dummies(data[col], prefix=col)
    df= pd.concat([df,y], axis=1)

data = data.drop(columns=categorical_columns)


data_1 = pd.concat([data,df], axis=1)

#Plot any two features (Age and Baseline_LDL) of the dataset. Later, we will plot these two features again, divided into the clusters DBSCAN gives
#in order to check how DBSCAN performs. 

data_1 = data_1.sort_values(by='Age')
plt.scatter(data_1.loc[:, 'Age'], data_1.loc[:, 'Baseline_LDL'])
plt.xlabel("Age")
plt.ylabel("Baseline_LDL")
plt.title("Before DBSCAN Clustering")
plt.show()


# #Normalize data
data_scaled = StandardScaler().fit_transform(data_1)

#Parameter Estimation/Evaluation for DBSCAN (eps and min_samples)

#For min_samples or min_points lets choose it to be equal to the number of columns of our transformed dataframe after encoding categorical variables
# Hence its value would be 46 

min_points= 46

print("Number of columns in transformed dataset: " + str(data_1.shape[1]))
print("Optimal Minimum Points (min_samples) are: " + str(min_points))

#ELBOW METHOD
#For eps or epsilon, the optimal value is found via Elbow Method. The average distance between each point and its k nearest neighbors is calculated, 
#where k equals the min_samples value we selected. The average k-distances once found are plotted in ascending order on a k-distance graph. 
#The best value for epsilon is at the point of maximum curvature (i.e. where the graph has the greatest slope) / elbow of the graph. 

#Find average distance between each point and its k nearest neighbor
n = NearestNeighbors(n_neighbors=min_points)

n_fit = n.fit(data_scaled)
distances, indices = n_fit.kneighbors(data_scaled)

#Sort in ascending order and plot the distances
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.xlabel('Distances')
plt.ylabel('Epsilion (eps)')
plt.title('k-distance elbow plot')
plt.plot(distances)
plt.show()

#From the graph we can see that at crook of the elbow, value of epsilon is around 6, which is our optimal value. 
#Now lets develop our Model since we have values of epsilon and min_samples. 


labels = DBSCAN(eps=6, min_samples=46).fit_predict(data_scaled)
#labels = db.labels_
print(labels)

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

print("Predicted Number of Distinct Clusters: " + str(realClusterNum))
print("Predicted Number of Distinct Clusters plus Noise Cluster (-1 labels): " + str(clusterNum))

#Hence we found total number of distinct clusters, excluding noise, to be 4. Now lets find the Silhouette Score.

print("Silhouette Coefficient for " + str(realClusterNum) + " clusters is: %0.2f"% silhouette_score(data_scaled, labels))



#Scatter Plot of Clusters 
fig, ax = plt.subplots()

scatter= ax.scatter(data_1.loc[:, 'Age'], data_1.loc[:, 'Baseline_LDL'], c=labels, cmap="plasma")
plt.xlabel("Age")
plt.ylabel("Baseline_LDL")
plt.title("After DBSCAN Clustering with " + str(realClusterNum) + " clusters")

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Clusters")
ax.add_artist(legend1)
plt.show()

#Conclusion
#Most of the data points have similar density and are therefore always grouped together in a single cluster, regardless of number of clusters.
#This is good since we checked that our provided labels are imbalanced, with most labels belonging to the class "Unlikely". However,
#since the 4 clusters do not appear to be seperated or distinct, DBSCAN has not worked well overall. Silhoutte Score is also less
#considering the data and number of clusters.  


filename = 'dbscan_model'
pickle.dump(model, open(filename,'wb'))
