import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# Convert the data string to a DataFrame
df = pd.read_csv("a.csv")

# Drop the 'Outcome' column
X = df.drop(['Outcome'], axis=1)


# Range of clusters to try
k_values = range(1, 15)

# Sum of squared distances for each k
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    clus_centre = kmeans.cluster_centers_

# Plot the Elbow Method graph
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.show()


####kmeans clustering dendogram###
Z = linkage(clus_centre, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram of K-means Clustering")
plt.xlabel("Cluster Index")
plt.ylabel("Distance")
plt.show()