import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score

data = pd.read_csv("a.csv")

x = data.drop("Outcome",axis=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
cluster_labels_using_kmeans=kmeans.labels_
print(cluster_labels_using_kmeans)


#Agglomerative heirarchical clustering ##
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage="ward")
cluster_labels_using_agglo = hierarchical_cluster.fit_predict(x)
print(cluster_labels_using_agglo)




##Silhoutte coefficient kmeans##
sample_silhouette_values = silhouette_samples(x, cluster_labels_using_kmeans)
silhouette_avg = silhouette_score(x, cluster_labels_using_kmeans)
print("Silhouette Scores:", sample_silhouette_values)
print("Average Silhouette Score:", silhouette_avg)

##Silhoutte coefficient hierachichal##
sample_silhouette_values = silhouette_samples(x, cluster_labels_using_agglo)
silhouette_avg = silhouette_score(x, cluster_labels_using_agglo)
print("Silhouette Scores (kmeans):", sample_silhouette_values)
print("Average Silhouette Score (kmeans):", silhouette_avg)
