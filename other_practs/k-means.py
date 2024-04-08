import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sb

#

# Convert the data string to a DataFrame
df = pd.read_csv("a.csv")

# Drop the 'Outcome' column
X = df.drop(['Outcome'], axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you can use the elbow method)
k = 3

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print(df['Cluster'])

print(df.head())

cluster_labels_1=kmeans.labels_
print(cluster_labels_1)

cluster_centers = kmeans.cluster_centers_
print(cluster_centers)

# # Visualize the clusters with a pair plot
# df_vis = df.drop(['Outcome'], axis=1)  # Exclude 'Outcome' column for visualization
# sns.pairplot(df_vis, hue='Cluster', palette='viridis')
# plt.suptitle('Pair Plot of Clusters', y=1.02)
# plt.show()


data = pd.read_csv("a.csv")
x = data.drop("Outcome",axis=1)

df = pd.DataFrame(x)
features =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


plt.figure(figsize=(9,6))
sb.scatterplot(x='Pregnancies',y='Glucose',hue=cluster_labels_1,palette = "Dark2", legend = False,data=x)
sb.scatterplot(x = cluster_centers[:, 0], y = cluster_centers[:, 1], color = "black", marker = "x", s = 200)
plt.ylabel('Glucose')
plt.xlabel('Pregnancies')
plt.title('Cluster Plot')
plt.show()
