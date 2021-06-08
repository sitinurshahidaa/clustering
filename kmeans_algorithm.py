import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

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
#Therefore our K Means clustering should ideally cluster most of the data points into one large cluster and other data points into other distinct but 
#smaller clusters.

#Drop Target Label Column 
data=data.drop(columns='DLCC Results')

#Convert categorical features using one-hot encoding. This is required because K Means does not work with categorical data.

categorical_columns  = ['Ethnic', 'Personal_PCAD', 'Gender', 'Personal_CVA_PVD', 'Smoking','Diabetes', 
'Family_Hypercholesterolaemia','Family_PCAD', 'Family_FH', 'On_LLT','Tendon_xanthomata', 'Corneal_arcus']

df = pd.DataFrame()
for col in categorical_columns:
    y = pd.get_dummies(data[col], prefix=col)
    df= pd.concat([df,y], axis=1)

data = data.drop(columns=categorical_columns)

data_1 = pd.concat([data,df], axis=1)

#Plot before applying k-means

data_1 = data_1.sort_values(by='Age')
plt.scatter(data_1.loc[:, 'Age'], data_1.loc[:, 'Baseline_LDL'])
plt.xlabel("Age")
plt.ylabel("Baseline_LDL")
plt.title("Before K-Means Clustering")
plt.show()



#Parameter Estimation/Evaluation for  K Means 

#Elbow Method

score = []

for cluster in range(1,11):
    kmeans = KMeans(n_clusters = cluster, init="k-means++", random_state=10)
    kmeans.fit(data_1)
    score.append(kmeans.inertia_)
    
# plotting the score

plt.plot(range(1,11), score)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()



# Silhouette score

for n_clusters in range(2,11):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data_1) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(data_1)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data_1, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is : %0.2f"% silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data_1, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(data_1.loc[:, 'Age'], data_1.loc[:, 'Baseline_LDL'], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

#Hence from both Elbow Method and Silhoutte Score optimal value of k appears to be 4. Now lets apply K means on our data and see a scatter 
#plot of clusters.

#Scatter Plot of Clusters 

model = KMeans(n_clusters=4, random_state=10)
labels = model.fit_predict(data_1)

fig, ax = plt.subplots()

scatter= ax.scatter(data_1.loc[:, 'Age'], data_1.loc[:, 'Baseline_LDL'], c=labels, cmap="plasma")
plt.xlabel("Age")
plt.ylabel("Baseline_LDL")
plt.title("After K Means Clustering with " + str(4) + " clusters")

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Clusters")
ax.add_artist(legend1)
plt.show()

#Conclusion
#K Means clustering with k=4 clusters has resulted in good, seperated clusters on our dataset. Silhoutte Score for k=4 is also reasonable 
# with the value being 0.57.  

#In comparison to DBSCAN which also had 4 optimal clusters, K Means provided more distinctly seperated clusters and also a higher Silhoutte Score. 
#Hence K Means proved to be better. 
